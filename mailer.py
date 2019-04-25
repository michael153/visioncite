# using SendGrid's Python Library
# https://github.com/sendgrid/sendgrid-python

import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

apikey='SG.ACAoABOJQZSJazaoxA11fw.gpelGQzsoBTgAukWu-WTjdQGjQdeLBbKtM1qAHo8Blk'

def send_email(subject, content):
	sender = "chtc-trainjob-noreply@eyecite.com"
	recipients = ["m.wan@berkeley.edu", "bveeramani@berkeley.edu"]
	for recipient in recipients:
		message = Mail(
		    from_email=sender,
		    to_emails=recipient,
		    subject=subject,
		    html_content=content)
		try:
		    sg = SendGridAPIClient(apikey)
		    response = sg.send(message)
		    print("Success sending email to %s" % recipient)
		except Exception as e:
		    print(e.message)
