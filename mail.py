"""Command-line tool for sending emails.

usage: mail.py [-h] [--subject SUBJECT] [--message MESSAGE] [--sender SENDER]
               recipient

Sends emails

positional arguments:
  recipient          recipient email address

optional arguments:
  -h, --help         show this help message and exit
  --subject SUBJECT  email subject line (default "Message from Eyecite")
  --message MESSAGE  the message to send (default "")
  --sender SENDER    sender email address (default "no-reply@eyecite.com")
"""
import argparse

from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

API_KEY = "SG.ACAoABOJQZSJazaoxA11fw.gpelGQzsoBTgAukWu-WTjdQGjQdeLBbKtM1qAHo8Blk"

DEFAULT_SENDER = "no-reply@eyecite.com"
DEFAULT_SUBJECT = "Message from Eyecite"
DEFAULT_MESSAGE = ""


def main():
    """Main function.

    Example:
        $ python mail.py --subject "Tennis Scores" --message \
          "Nadal wins 6-0 6-0." --sender no-reply@tennis.com \
          bveeramani@berkeley.edu
    """
    parser = argparse.ArgumentParser(description='Sends email')
    parser.add_argument('recipient', help='recipient email address')
    parser.add_argument(
        '--subject',
        dest="subject",
        default=DEFAULT_SUBJECT,
        help='email subject line (default "Message from Eyecite")')
    parser.add_argument('--message',
                        dest="message",
                        default=DEFAULT_MESSAGE,
                        help='the message to send (default "")')
    parser.add_argument(
        '--sender',
        dest="sender",
        default=DEFAULT_SENDER,
        help='sender email address (default "no-reply@eyecite.com")')

    args = parser.parse_args()

    try:
        send_email(args.subject, args.message, args.recipient, args.sender)
        return 0
    except Exception:  # pylint: disable=broad-except
        return 1


def send_email(subject, message, recipient, sender=DEFAULT_SENDER):
    """Sends an email using the SendGrid API."""
    mail = Mail(from_email=sender,
                to_emails=recipient,
                subject=subject,
                html_content=message)
    client = SendGridAPIClient(API_KEY)
    client.send(mail)


if __name__ == "__main__":
    main()
