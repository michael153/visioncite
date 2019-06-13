import argparse
import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

API_KEY = "SG.ACAoABOJQZSJazaoxA11fw.gpelGQzsoBTgAukWu-WTjdQGjQdeLBbKtM1qAHo8Blk"
DEFAULT_SENDER = "no-reply@eyecite.com"


def main():
    parser = argparse.ArgumentParser(description='Send mail')
    parser.add_argument('subject',
                        help='email subject line')
    parser.add_argument('message',
                        help='the message to send')
    parser.add_argument('recipient',
                        help='recipient email address')
    parser.add_argument('--sender',
                        dest="sender",
                        default=DEFAULT_SENDER,
                        help='sender email address')

    args = parser.parse_args()

    try:
        send_email(args.subject, args.message, args.recipient, args.sender)
        return 0
    except Exception as e:
        return 1


def send_email(subject, message, recipient, sender=DEFAULT_SENDER):
    mail = Mail(from_email=sender,
                to_emails=recipient,
                subject=subject,
                html_content=message)
    client = SendGridAPIClient(API_KEY)
    response = client.send(mail)


if __name__ == "__main__":
    main()
