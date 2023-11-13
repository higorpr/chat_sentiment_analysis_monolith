import random

from flask import Blueprint, jsonify, request
from app.services.report_service import ReportService
from app.exceptions.errors import (
    InexistantChat,
    VoidChatHistory,
    NoClientMessages,
    S3UploadError,
)
from bson.errors import InvalidId
from datetime import datetime as dt, timedelta
from dateutil import parser

report_blueprint = Blueprint("chat_sentiment", __name__, url_prefix="/report")


@report_blueprint.route("/health", methods=["GET"])
def return_health():
    return 'OK', 200


@report_blueprint.route("/", methods=["GET"])
def getSentimentReport():
    report_service = ReportService()

    # Retrieve account_id and wa_chat_id from query string:
    account_id = request.args.get("account_id")
    wa_chat_id = request.args.get("wa_chat_id")

    # Check if account_id and wa_chat_id are present:
    if account_id == None or wa_chat_id == None:
        return (
            jsonify(
                {
                    "error": "Account ID (account_id) and Whatsapp Chat ID(wa_chat_id) are required"
                }
            ),
            400,
        )

    # Retrieve chat_id:
    try:
        chat_id = report_service.get_chat_id(account_id, wa_chat_id)
    except InexistantChat as err:
        return str(err), 404
    except InvalidId as err:
        return str(err), 400

    # Retrieve messages:
    messages = report_service.get_chat_messages(chat_id)

    # Create messages dataframe:
    messages_df = report_service.import_data(messages)

    # Data Processing:

    # Remove non-client data:
    client_messages_df = report_service.message_cleanup(messages_df)

    # Model Application:

    # Apply LeIA model:
    classified_messages_df = report_service.chat_classification(client_messages_df)

    # Calculate messages' weights:
    try:
        weighted_df = report_service.generate_weighted_df(classified_messages_df)
    except NoClientMessages as err:
        return str(err), 404

    # Generate whole chat sentiment:
    coefficient = report_service.calculate_chat_sentiment_coef(weighted_df)

    sat_label = report_service.generate_sentiment_label(coefficient)

    # # Format Chat Messages:
    # formated_chat = report_service.format_chat(messages)
    # Create pdf byte file:
    # pdf_buffer = report_service.create_report(formated_chat, coefficient)

    # # S3 upload:
    # s3_path = f"report/sentiment_analysis_report.pdf"
    # try:
    #     report_service.update_file_to_s3(pdf_buffer, s3_path)
    # except S3UploadError as err:
    #     return str(err), 500

    # Return:
    return f'The satisfaction label for the calculated coefficient is "{sat_label}"!', 200


@report_blueprint.route("/joint_sentiment", methods=["GET"])
def join_sentiment_coefficients():

    # Retrieve date limits using query strings:
    from_date = request.args.get("from_date")
    to_date = request.args.get("to_date")
    print(from_date)

    # Check if account_id and wa_chat_id are present:
    if from_date == None or to_date == None:
        return (
            jsonify(
                {
                    "error": "Limiting dates are necessary to generate a sentiment coefficient"
                }
            ),
            400,
        )

    current_day = parser.parse(from_date)
    last_day = parser.parse(to_date)

    count = 0
    coef = 0
    while current_day <= last_day:
        # TODO: Remover geração automática de coeficientes e adicionar busca por SentimentSnapshots
        coef += random.randint(-2, 2)
        count += 1
        current_day = current_day + timedelta(days=1)

    coef = coef / count

    return str(coef), 200
