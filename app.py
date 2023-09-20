import os

from aws_lambda_powertools.event_handler import APIGatewayRestResolver
from aws_lambda_powertools.utilities.typing import LambdaContext

from errors import InexistantChat, VoidChatHistory, NoClientMessages, S3UploadError
from bson.errors import InvalidId

from functions import (
    chat_id_verification,
    get_chat_messages,
    import_data,
    message_cleanup,
    chat_classification,
    generate_weighted_df,
    calculate_chat_sentiment_coef,
    generate_sentiment_label,
    format_chat,
    create_report,
    update_file_to_s3,
)

# App Creation
app = APIGatewayRestResolver()


@app.get("/chat/<chat_id>")
def hello_name(chat_id):
    # Check chat id:
    try:
        chat_id_verification(chat_id)

    except InexistantChat as err:
        return str(err), 404
    except InvalidId as err:
        return str(err), 400

    # Retrieve messages:
    try:
        messages = get_chat_messages(chat_id)
    except VoidChatHistory as err:
        return str(err), 404

    # Create messages dataframe:
    messages_df = import_data(messages)

    ## Data Processing ##

    # Remove non-client data:
    client_messages_df = message_cleanup(messages_df)

    # Model Application ##

    # Apply LeIA model:
    classified_messages_df = chat_classification(client_messages_df)

    ## Chat Sentiment Coefficient Calculation ##

    # Calculate messages' weights:
    try:
        weighted_df = generate_weighted_df(classified_messages_df)
    except NoClientMessages as err:
        return str(err), 404

    # Generate whole chat sentiment:
    coefficient = calculate_chat_sentiment_coef(weighted_df)

    sat_label = generate_sentiment_label(coefficient)

    # Format Chat Messages
    formated_chat = format_chat(messages)
    # Create pdf byte file
    pdf_buffer = create_report(formated_chat, coefficient)

    # # S3 Variables
    s3_bucket = os.environ[os.environ.get("AWS", "BUCKET_NAME")]
    s3_path = f"report/sentiment_analysis_report-{chat_id}.pdf"

    # Upload data to S3
    try:
        update_file_to_s3(pdf_buffer, s3_bucket, s3_path)
    except S3UploadError as err:
        return str(err), 500
    return {
        "statusCode": 200,
        "body": f'The satisfaction label for the calculated coefficient is "{sat_label}"!',
    }


def lambda_handler(event: dict, context: LambdaContext) -> dict:
    return app.resolve(event, context)
