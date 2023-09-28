from flask import Blueprint, jsonify, request
from app.services.report_service import ReportService
from app.exceptions.errors import (
    InexistantChat,
    VoidChatHistory,
    NoClientMessages,
    S3UploadError,
)
from bson.errors import InvalidId

report_blueprint = Blueprint("report", __name__, url_prefix="/report")


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
    return jsonify(
        {
            "statusCode": 200,
            "body": f'The satisfaction label for the calculated coefficient is "{sat_label}"!',
        },
        200,
    )
