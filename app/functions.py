import pandas as pd
import emoji
import boto3

from app.exceptions.errors import (
    InexistantChat,
    VoidChatHistory,
    NoClientMessages,
    S3UploadError,
)
from app.database.connection import messages_db, chats_db
from bson.objectid import ObjectId
from io import BytesIO
from LeIA import SentimentIntensityAnalyzer as LeiaAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as EmojiAnalyzer
from app.database.s3_connection import s3_bucket

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    ListFlowable,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle as PS


def chat_id_verification(chat_id: str):
    chat_check = chats_db.find_one({"_id": ObjectId(chat_id)})

    if chat_check == None:
        raise InexistantChat(f"Erro: Arquivo do chat ${chat_id} não existe.")


def get_chat_messages(chat_id: str):
    messages = []
    query = {
        "chat": ObjectId(chat_id),
        "type": "chat",
        "text": {"$exists": "true"},
    }

    messages = messages_db.find(query).sort("send_date", 1)
    messages = list(messages)

    if len(messages) < 3:
        raise VoidChatHistory(
            "Chat não tem mensagens suficientes para uma análise de sentimento"
        )

    return messages


def import_data(messages: list):
    order = 1

    messages_dict = {
        "id": [],
        "text": [],
        "source": [],
        "send_date": [],
        "order_in_chat": [],
    }

    for m in messages:
        # Get message id
        messages_dict["id"].append(m["_id"])

        # Get message source
        if m["is_out"]:
            messages_dict["source"].append("A")
            # Add order in chat sequence for Attendant:
            messages_dict["order_in_chat"].append("NA")
        else:
            messages_dict["source"].append("C")
            # Add order in chat sequence for Client:
            messages_dict["order_in_chat"].append(order)
            order += 1

        # Get message text
        messages_dict["text"].append(m["text"])

        # Get message datetime
        messages_dict["send_date"].append(m["timestamp"])

    messages_df = pd.DataFrame(data=messages_dict)

    # Sort messages by send_date
    messages_df.sort_values(by=["send_date"], inplace=True)

    return messages_df


def message_cleanup(messages_df):
    cleaned_messages_df = messages_df[messages_df["source"] == "C"]
    cleaned_messages_df.reset_index(drop=True, inplace=True)

    return cleaned_messages_df


# Function to normalize LeIA compounds:
def extract_leia_sentiment(compound):
    sent_output = {"label": "", "new_score": 0}

    sent_output["new_score"] = (compound + 1) / 2

    if compound == 0:
        sent_output["label"] = 0
    elif compound > 0.2:
        sent_output["label"] = 2
    elif compound > 0:
        sent_output["label"] = 1
    elif compound >= -0.2:
        sent_output["label"] = -1
    else:
        sent_output["label"] = -2

    return sent_output


def split_message_sections(message: str):
    emoji_list = emoji.emoji_list(message)
    emojis = ""
    text = message

    # Case where the message does not have emojis
    if len(emoji_list) == 0:
        return {"text": text, "emojis": emojis}

    for e in emoji_list:
        # Add emoji to emoji list
        emojis += e["emoji"]

        # Remove emoji from text
        text = text.replace(e["emoji"], "")

    return {"text": text, "emojis": emojis}


def get_message_compound(message: str) -> float:
    leia = LeiaAnalyzer()
    vader = EmojiAnalyzer()

    split_message = split_message_sections(message)

    # Get text message compound
    text_compound = leia.polarity_scores(split_message["text"])["compound"]

    # Get emojis compound
    emoji_compound = vader.polarity_scores(split_message["emojis"])["compound"]

    message_compound = round((text_compound + 2 * emoji_compound) / 3, 4)

    return message_compound


# LeIA Method Function
def chat_classification(messages_df):
    # Verificação de emojis

    # Apply leia classifier
    classified_df = messages_df.assign(
        score=messages_df["text"].apply(lambda x: get_message_compound(x))
    )

    # Generate labels and normalized classification score
    classified_df = classified_df.assign(
        classification_score=classified_df["score"].apply(
            lambda x: extract_leia_sentiment(x)["new_score"]
        ),
        classification_label=classified_df["score"].apply(
            lambda x: extract_leia_sentiment(x)["label"]
        ),
    )

    # Remove unecessary columns
    classified_df.drop(columns=["score", "source"], inplace=True)

    return classified_df


# Function to calculate individual message weight:
def calculate_weight(order: int, n_messages: int):
    if n_messages < 1:
        raise NoClientMessages("Não há mensagens de clientes no chat")
    den = 0
    for i in range(1, n_messages + 1):
        den += i**2
    w = (order**2) / den

    return w


# Function to generate a dataframe with weighted messages:
def generate_weighted_df(df: pd.DataFrame):
    n_messages = df.shape[0]
    df = df.assign(
        message_weight=df.apply(
            lambda x: calculate_weight(x["order_in_chat"], n_messages), axis=1
        )
    )

    return df


# Function to calculate chat sentiment based on message weights and classification score
def calculate_chat_sentiment_coef(df):
    num = 0
    den = 0
    for idx, row in df.iterrows():
        num += row["classification_label"] * row["message_weight"]
        den += row["message_weight"]
    coef = num / den
    return coef


# Function to generate the satisfaction label of the chat:
def generate_sentiment_label(coef: float):
    label = ""
    if coef > 2 or coef < -2:
        return "Houve um erro, por favor entre em contato com o suporte da ChatGuru"

    if coef <= -1:
        label = "Insatisfeito"
    elif coef < -0.2:
        label = "Levemente Insatisfeito"
    elif coef <= 0.2:
        label = "Neutro"
    elif coef < 1:
        label = "Levemente Satisfeito"
    else:
        label = "Satisfeito"

    return label


def format_chat(messages: list):
    chat_text = []

    for m in messages:
        # Get message source
        if m["is_out"]:
            attendant_message = f"[Atendente] ({m['timestamp']}) {m['text']}"
            chat_text.append(attendant_message)
        else:
            client_message = f"[Cliente] ({m['timestamp']}) {m['text']}"
            chat_text.append(client_message)

    return chat_text


def create_report(formated_chat: list, sentiment_coef: float):
    # Create byte buffer to hold report information
    pdf_buffer = BytesIO()
    # Create document
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    # Report element list
    elements = []
    # Styles Class Instance
    styles = getSampleStyleSheet()
    # Creates a style for centered text
    centered_style = PS(name="CenteredStyle", parent=styles["Heading3"], alignment=1)

    # TITLE
    title = Paragraph("Relatório de Análise de Sentimento do Chat", styles["Title"])
    elements.append(title)
    elements.append(Spacer(1, 20))

    # SECTION - "Method for Sentiment Analysis"
    section_title = Paragraph("Método para Análise de Sentimento", styles["Heading1"])
    elements.append(section_title)

    # SUBSECTION - "Method Description"
    subtitle = Paragraph("Descrição do Método", styles["Heading2"])
    elements.append(subtitle)
    elements.append(Spacer(1, 10))

    # TEXT - "method description"
    method_introduction = (
        "O método para a obtenção da estimativa do sentimento de um cliente durante "
        "uma interação com o atendimento consiste em :"
    )

    text = Paragraph(method_introduction, styles["Normal"])
    elements.append(text)
    elements.append(Spacer(1, 10))

    method_list = [
        "Análise do sentimento de todas as mensagens dos clientes",
        "Cálculo do peso de cada mensagem",
        "Cálculo da média do sentimento do chat completo",
        "Interpretação do resultado da análise",
    ]

    numbered_list = ListFlowable(
        [
            Paragraph(f"{item}", styles["Normal"])
            for i, item in enumerate(method_list, start=1)
        ],
        bulletType="bullet",
        leftIndent=20,
    )
    elements.append(numbered_list)
    elements.append(Spacer(1, 10))

    entry_1 = (
        "O primeiro passo consiste na aplicação do modelo de aprendizado de máquina treinado para a "
        "classificação do sentimento do cliente em cada uma das mensagens enviadas para o atendente, gerando assim "
        'um nível estimado de satisfação do cliente que varia entre "Satisfeito", "Levemente Satisfeito", "Neutro"'
        ', "Levemente Insatisfeito" ou "Insatisfeito".'
    )

    text = Paragraph(entry_1, styles["Normal"])
    elements.append(text)
    elements.append(Spacer(1, 10))

    entry_2 = (
        "O que se segue é a transformação das classificações dos sentimentos individuais "
        "expressos em cada uma das mensagens em pesos matemáticos que compõem o sentimento do cliente "
        "durante todo o atendimento. Esses pesos são definidos seguindo-se a metodologia formulada internamente"
        " pelo time de Inteligência Artificial da ChatGuru."
    )

    text = Paragraph(entry_2, styles["Normal"])
    elements.append(text)
    elements.append(Spacer(1, 10))

    entry_3 = (
        "Usando-se parâmetros obtidos do chat completo e do modelo de IA da ChatGuru, é calculado um "
        "coeficiente numérico de satisfação do atendimento completo."
    )

    text = Paragraph(entry_3, styles["Normal"])
    elements.append(text)
    elements.append(Spacer(1, 10))

    entry_4 = (
        "Por fim, esse coeficiente de satisfação é interpretado em termos não-matemáticos para ser "
        "apreciado pelo contratante do serviço."
    )

    text = Paragraph(entry_4, styles["Normal"])
    elements.append(text)
    elements.append(Spacer(1, 20))
    elements.append(PageBreak())

    # SECTION - "Sentiment Analysis"
    section_title = Paragraph("Análise de Sentimento", styles["Heading1"])
    elements.append(section_title)

    # SUBSECTION - "Chat Presentation"
    subtitle = Paragraph("Apresentação do Chat", styles["Heading2"])
    elements.append(subtitle)
    elements.append(Spacer(1, 10))

    # TEXT - "chat content"
    for line in formated_chat:
        chat = Paragraph(line, styles["Normal"])
        elements.append(chat)
        elements.append(Spacer(1, 5))

    # SUBSECTION - "Analysis result"
    subtitle = Paragraph("Resultados da Análise", styles["Heading2"])
    elements.append(subtitle)
    elements.append(Spacer(1, 10))

    # TEXT - "analysis result intro"
    analysis_result = (
        "Ao se aplicar o método já descrito neste relatório, o coeficiente de satisfação do usuário na "
        "conversa apresentada como objeto de análise foi de:"
    )
    text = Paragraph(analysis_result, styles["Normal"])
    elements.append(text)
    elements.append(Spacer(1, 8))

    # TEXT - "sentiment coefficient"
    str_coef = str(round(sentiment_coef, 3))
    text = f"coeficiente de satisfação = {str_coef}"
    centered_text = Paragraph(text, centered_style)
    elements.append(centered_text)
    elements.append(Spacer(1, 10))

    # TEXT - "result interpretation"
    sentiment_label = generate_sentiment_label(sentiment_coef)
    interpretation = f"Dado o coeficiente de satisfação apresentado, podemos estimar que o cliente se sentiu:"
    text = Paragraph(interpretation, styles["Normal"])
    elements.append(text)

    centered_text = Paragraph(sentiment_label, centered_style)
    elements.append(centered_text)
    elements.append(Spacer(1, 10))

    # Build the rest of the report
    doc.build(elements)

    # Move buffer position to the beginning
    pdf_buffer.seek(0)

    return pdf_buffer


def update_file_to_s3(data, s3_path):
    s3 = boto3.client("s3")
    try:
        s3.upload_fileobj(data, s3_bucket, s3_path)
    except Exception:
        raise S3UploadError(
            "Houve um erro ao fazer o upload do arquivo para o bucket S3"
        )
