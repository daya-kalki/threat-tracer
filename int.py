import email
from email import policy

# Path to the .eml file
eml_file = "sample.eml"

# Open and parse the .eml file
with open(eml_file, "rb") as file:
    msg = email.message_from_bytes(file.read(), policy=policy.default)

# Extract email details
subject = msg["Subject"]
sender = msg["From"]
recipient = msg["To"]

# Extract email body
body = ""
if msg.is_multipart():
    for part in msg.walk():
        content_type = part.get_content_type()
        if content_type == "text/plain":  # Extract plain text content
            body = part.get_payload(decode=True).decode()
            break
else:
    body = msg.get_payload(decode=True).decode()

# Print the extracted content
print("Subject:", subject)
print("From:", sender)
print("To:", recipient)
print("Body:", body)
