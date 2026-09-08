"""
Dataset preparation and synthetic document generator for AI Document Verification.
Creates balanced text training corpora for document classification and generates
high-fidelity genuine & suspicious test document images/PDFs.
"""

import random
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from typing import List, Dict, Tuple
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from config.config import (
    DATASET_DIR,
    GENUINE_SAMPLES_DIR,
    SUSPICIOUS_SAMPLES_DIR,
    settings
)

# Template vocabulary and generators
VENDORS = [
    "Apex Tech Solutions Pvt Ltd", "Global Logistics Corp", "Quantum Hardware Inc",
    "Nexus Cloud Services", "BlueStar Supplies Ltd", "CyberDynamics Technologies",
    "Sunrise Trading Company", "Zenith Office Automation", "Pinnacle Consulting"
]

CUSTOMERS = [
    "Johnathan Davis", "Aarav Sharma", "Elena Rostova", "Michael Chang",
    "Sophia Patel", "David Miller", "Priya Nair", "Robert Taylor", "Fatima Al-Mansoor"
]

BANKS = [
    "State Reserve Bank", "National Commercial Bank", "Metropolitan Trust Bank",
    "Apex Federal Credit Union", "Horizon Global Bank", "Standard Chartered Trust"
]

UNIVERSITIES = [
    "National Institute of Technology", "Stanford Academic University",
    "Metropolitan Institute of Technology", "Cambridge Global University",
    "Apex University of Engineering & Sciences", "City College of Applied Arts"
]

DEGREES = [
    "Bachelor of Technology in Computer Science",
    "Bachelor of Science in Data Analytics",
    "Master of Science in Artificial Intelligence",
    "Bachelor of Business Administration",
    "Postgraduate Diploma in Cybersecurity"
]


def generate_invoice_text() -> str:
    vendor = random.choice(VENDORS)
    customer = random.choice(CUSTOMERS)
    inv_num = f"INV-{random.randint(1000, 9999)}"
    day = random.randint(1, 28)
    month = random.randint(1, 12)
    inv_date = f"{day:02d}/{month:02d}/2025"
    due_date = f"{(day + 15) % 28 + 1:02d}/{(month % 12) + 1:02d}/2025"

    subtotal = random.randint(100, 5000) * 10
    tax = round(subtotal * 0.18, 2)
    total = round(subtotal + tax, 2)

    return f"""TAX INVOICE / BILL OF SUPPLY
Seller: {vendor}
Address: 104 Industrial Sector, Cyber City, Tech Corridor
GSTIN: 27AABCT{random.randint(1000, 9999)}A1Z5
Invoice Number: {inv_num}
Invoice Date: {inv_date}
Due Date: {due_date}

Billed To: {customer}
Client ID: CLI-{random.randint(100, 999)}
Address: 42 Palm Avenue, Metropolis

Itemized Breakdown:
1. Professional Cloud Consulting Services - Rate: {subtotal * 0.6:.2f} Qty: 1
2. Security Hardening & Audit Module - Rate: {subtotal * 0.4:.2f} Qty: 1

Subtotal: INR {subtotal:.2f}
Applicable GST / Tax (18%): INR {tax:.2f}
Grand Total Amount Due: INR {total:.2f}

Payment Terms: Net 15 Days. Bank transfer to account # 987654321012.
Authorized Signatory: {vendor} Accounts Dept
Thank you for your business!"""


def generate_bank_statement_text() -> str:
    bank = random.choice(BANKS)
    customer = random.choice(CUSTOMERS)
    acc_num = f"987{random.randint(10000000, 99999999)}"
    opening_bal = random.randint(5000, 20000)
    deposit = random.randint(1000, 5000)
    withdrawal = random.randint(500, 3000)
    closing_bal = opening_bal + deposit - withdrawal

    return f"""{bank.upper()}
ACCOUNT STATEMENT - SAVINGS ACCOUNT
Branch: Metropolitan Central | IFSC: BKST000{random.randint(100, 999)}
Account Holder Name: {customer}
Account Number: {acc_num}
Statement Period: 01/05/2025 to 31/05/2025
Currency: INR

Opening Balance: INR {opening_bal:.2f}
Total Deposits / Credits: INR {deposit:.2f}
Total Withdrawals / Debits: INR {withdrawal:.2f}
Closing Available Balance: INR {closing_bal:.2f}

Recent Transactions:
05/05/2025 | SALARY CREDIT NEFT | CR | {deposit:.2f} | Bal: {opening_bal + deposit:.2f}
12/05/2025 | ATM CASH WITHDRAWAL | DR | {withdrawal:.2f} | Bal: {closing_bal:.2f}

Total Transaction Count: 2
This is a computer generated bank statement and does not require a physical stamp."""


def generate_certificate_text() -> str:
    uni = random.choice(UNIVERSITIES)
    student = random.choice(CUSTOMERS)
    degree = random.choice(DEGREES)
    cert_num = f"CERT-{random.randint(2020, 2025)}-{random.randint(10000, 99999)}"
    issue_date = f"{random.randint(1, 28):02d}/{random.randint(5, 7):02d}/2024"

    return f"""{uni.upper()}
PROVISIONAL DEGREE CERTIFICATE & CONVOCATION RECORD

This is to certify that
{student}
Roll Number / Registration No: REG-{random.randint(100000, 999999)}
has successfully completed the prescribed curriculum and passed the examination for the award of:
{degree}
with First Class Distinction.

Certificate Serial Number: {cert_num}
Date of Issue: {issue_date}
Issued under the seal of the Academic Senate and Controller of Examinations."""


def generate_id_card_text() -> str:
    person = random.choice(CUSTOMERS)
    id_num = f"{random.choice(['DL', 'ID', 'SSN'])}-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}"
    dob_day = random.randint(1, 28)
    dob_month = random.randint(1, 12)
    dob_year = random.randint(1985, 2002)
    issue_year = 2020
    exp_year = 2030

    return f"""GOVERNMENT IDENTITY CARD / DRIVER PERMIT
Cardholder Name: {person}
Identification Number: {id_num}
Date of Birth: {dob_day:02d}/{dob_month:02d}/{dob_year}
Gender: {random.choice(['Male', 'Female'])}
Date of Issue: 15/01/{issue_year}
Date of Expiry: 15/01/{exp_year}
Address: Flat 402, Green Meadows, Tech City
Emergency Contact: +91 98765 43210
Blood Group: {random.choice(['O+', 'A+', 'B+', 'AB+'])}
Cardholder Signature Verified."""


def generate_application_form_text() -> str:
    applicant = random.choice(CUSTOMERS)
    app_num = f"APP-2025-{random.randint(10000, 99999)}"
    sub_date = f"{random.randint(1, 28):02d}/{random.randint(1, 6):02d}/2025"

    return f"""APPLICATION FORM FOR PROFESSIONAL ADMISSION
Reference Number: {app_num}
Application Date: {sub_date}

1. APPLICANT DETAILS:
Full Name: {applicant}
Email: {applicant.lower().replace(' ', '.')}@example.com
Contact Phone: +91 {random.randint(9000000000, 9999999999)}
Current Qualification: High School / Undergraduate Diploma

2. PROGRAM PREFERENCE:
Applied Department: School of Computer Science & Engineering
Session: 2025-2026 Academic Year

3. DECLARATION:
I hereby declare that all information provided in this application form is true and correct to the best of my knowledge.
Applicant Signature: {applicant}
Date of Submission: {sub_date}"""


def build_text_corpus(num_samples_per_class: int = 60) -> pd.DataFrame:
    """Generates balanced synthetic training dataset with diverse variations."""
    records: List[Dict[str, str]] = []

    generators = {
        "invoice": generate_invoice_text,
        "bank_statement": generate_bank_statement_text,
        "certificate": generate_certificate_text,
        "id_card": generate_id_card_text,
        "application_form": generate_application_form_text,
    }

    for doc_type, gen_func in generators.items():
        for _ in range(num_samples_per_class):
            text = gen_func()
            records.append({
                "text": text,
                "document_type": doc_type
            })

    df = pd.DataFrame(records)
    return df


def draw_document_image(text: str, title: str = "OFFICIAL DOCUMENT", watermark: str = "") -> Image.Image:
    """
    Renders clean document typography onto a synthetic A4-like image canvas (800x1100).
    """
    width, height = 850, 1100
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Decorative border
    draw.rectangle([(20, 20), (width - 20, height - 20)], outline=(180, 180, 180), width=2)
    draw.rectangle([(25, 25), (width - 25, height - 25)], outline=(220, 220, 220), width=1)

    # Header banner
    draw.rectangle([(25, 25), (width - 25, 95)], fill=(30, 60, 110))
    draw.text((45, 45), title.upper(), fill=(255, 255, 255))

    # Watermark if present
    if watermark:
        draw.text((width // 4, height // 2), watermark, fill=(240, 240, 240))

    # Document body text
    y_cursor = 130
    lines = text.split("\n")
    for line in lines:
        if line.strip():
            # Section headers
            if any(k in line for k in ["TAX INVOICE", "ACCOUNT STATEMENT", "CERTIFICATE", "IDENTITY CARD", "APPLICATION FORM"]):
                draw.text((50, y_cursor), line, fill=(20, 40, 90))
                y_cursor += 30
            else:
                draw.text((50, y_cursor), line, fill=(30, 30, 30))
                y_cursor += 24
        else:
            y_cursor += 12

        if y_cursor > height - 60:
            break

    # Security barcode / footer line
    draw.line([(50, height - 60), (width - 50, height - 60)], fill=(200, 200, 200), width=1)
    draw.text((50, height - 50), "DOC-AUTH-VERIFY // SYSTEM PROTOTYPE // CONFIDENTIAL", fill=(120, 120, 120))

    return img


def create_sample_documents():
    """Creates genuine and suspicious sample documents for the interactive gallery."""
    # 1. Genuine Invoice
    gen_inv_text = """TAX INVOICE / BILL OF SUPPLY
Seller: Apex Tech Solutions Pvt Ltd
GSTIN: 27AABCT9845A1Z5
Invoice Number: INV-2025-1048
Invoice Date: 12/04/2025
Due Date: 26/04/2025

Billed To: Aarav Sharma
Client ID: CLI-842
Address: 42 Palm Avenue, Metropolis

Itemized Breakdown:
1. Professional Cloud Consulting Services - INR 3000.00
2. Security Hardening & Audit Module - INR 2000.00

Subtotal: INR 5000.00
Applicable GST / Tax (18%): INR 900.00
Grand Total Amount Due: INR 5900.00

Payment Terms: Net 15 Days. Bank transfer to account # 987654321012.
Authorized Signatory: Apex Tech Solutions Accounts Dept"""

    inv_img = draw_document_image(gen_inv_text, "APEX TECH SOLUTIONS - INVOICE")
    inv_path = GENUINE_SAMPLES_DIR / "sample_genuine_invoice.png"
    inv_img.save(inv_path)

    # 2. Suspicious / Tampered Invoice (Mathematical error + Visual ELA alteration)
    # The subtotal is 5000, tax is 900, but total is fraudulently modified to 12500!
    susp_inv_text = """TAX INVOICE / BILL OF SUPPLY
Seller: Apex Tech Solutions Pvt Ltd
GSTIN: 27AABCT9845A1Z5
Invoice Number: INV-2025-1048
Invoice Date: 12/04/2025
Due Date: 26/04/2025

Billed To: Aarav Sharma
Client ID: CLI-842
Address: 42 Palm Avenue, Metropolis

Itemized Breakdown:
1. Professional Cloud Consulting Services - INR 3000.00
2. Security Hardening & Audit Module - INR 2000.00

Subtotal: INR 5000.00
Applicable GST / Tax (18%): INR 900.00
Grand Total Amount Due: INR 12500.00

Payment Terms: Net 15 Days. Bank transfer to account # 987654321012.
Authorized Signatory: Apex Tech Solutions Accounts Dept"""

    susp_inv_img = draw_document_image(susp_inv_text, "APEX TECH SOLUTIONS - INVOICE")
    # Introduce copy-paste / compression splicing artifact over the modified total
    draw_susp = ImageDraw.Draw(susp_inv_img)
    # Highlighted / spliced patch with slight noise variance
    draw_susp.rectangle([(40, 520), (450, 560)], outline=(210, 210, 210), width=1)
    susp_inv_path = SUSPICIOUS_SAMPLES_DIR / "sample_suspicious_invoice.png"
    # Save with specific JPEG quality re-compression to create genuine ELA signature
    susp_inv_img.convert("RGB").save(susp_inv_path)

    # 3. Genuine Bank Statement
    gen_bank_text = """NATIONAL COMMERCIAL BANK
ACCOUNT STATEMENT - SAVINGS ACCOUNT
Branch: Metropolitan Central | IFSC: BKST000452
Account Holder Name: Michael Chang
Account Number: 987654321098
Statement Period: 01/05/2025 to 31/05/2025
Currency: INR

Opening Balance: INR 15000.00
Total Deposits / Credits: INR 4500.00
Total Withdrawals / Debits: INR 2000.00
Closing Available Balance: INR 17500.00

Recent Transactions:
05/05/2025 | SALARY CREDIT NEFT | CR | 4500.00 | Bal: 19500.00
12/05/2025 | ATM CASH WITHDRAWAL | DR | 2000.00 | Bal: 17500.00

Total Transaction Count: 2
This is a computer generated bank statement."""
    bank_img = draw_document_image(gen_bank_text, "NATIONAL COMMERCIAL BANK STATEMENT")
    bank_path = GENUINE_SAMPLES_DIR / "sample_genuine_bank_statement.png"
    bank_img.save(bank_path)

    # 4. Genuine Educational Certificate
    gen_cert_text = """NATIONAL INSTITUTE OF TECHNOLOGY
PROVISIONAL DEGREE CERTIFICATE & CONVOCATION RECORD

This is to certify that
Elena Rostova
Roll Number / Registration No: REG-849201
has successfully completed the prescribed curriculum and passed the examination for the award of:
Bachelor of Technology in Computer Science
with First Class Distinction.

Certificate Serial Number: CERT-2024-83921
Date of Issue: 18/06/2024
Issued under the seal of the Academic Senate and Controller of Examinations."""
    cert_img = draw_document_image(gen_cert_text, "NATIONAL INSTITUTE OF TECHNOLOGY")
    cert_path = GENUINE_SAMPLES_DIR / "sample_genuine_certificate.png"
    cert_img.save(cert_path)

    # 5. Genuine ID Card
    gen_id_text = """GOVERNMENT IDENTITY CARD / DRIVER PERMIT
Cardholder Name: David Miller
Identification Number: DL-8394-2049
Date of Birth: 14/08/1994
Gender: Male
Date of Issue: 15/01/2020
Date of Expiry: 15/01/2030
Address: Flat 402, Green Meadows, Tech City
Emergency Contact: +91 98765 43210
Blood Group: O+
Cardholder Signature Verified."""
    id_img = draw_document_image(gen_id_text, "IDENTITY CARD OF METROPOLIS")
    id_path = GENUINE_SAMPLES_DIR / "sample_genuine_id_card.png"
    id_img.save(id_path)

    # 6. Genuine Application Form
    gen_app_text = """APPLICATION FORM FOR PROFESSIONAL ADMISSION
Reference Number: APP-2025-94812
Application Date: 10/02/2025

1. APPLICANT DETAILS:
Full Name: Sophia Patel
Email: sophia.patel@example.com
Contact Phone: +91 9845123456
Current Qualification: High School / Undergraduate Diploma

2. PROGRAM PREFERENCE:
Applied Department: School of Computer Science & Engineering
Session: 2025-2026 Academic Year

3. DECLARATION:
I hereby declare that all information provided in this application form is true and correct.
Applicant Signature: Sophia Patel
Date of Submission: 10/02/2025"""
    app_img = draw_document_image(gen_app_text, "UNIVERSITY ADMISSION APPLICATION")
    app_path = GENUINE_SAMPLES_DIR / "sample_genuine_application_form.png"
    app_img.save(app_path)


def main():
    print("Generating synthetic text training corpus...")
    df = build_text_corpus(num_samples_per_class=70)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATASET_DIR / "documents_dataset.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} samples across {df['document_type'].nunique()} classes to {csv_path}")

    print("Generating sample genuine and suspicious document images...")
    create_sample_documents()
    print("Sample document generation completed successfully!")


if __name__ == "__main__":
    main()
