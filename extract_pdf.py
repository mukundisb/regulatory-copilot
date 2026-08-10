import pdfplumber

with pdfplumber.open("EU-MDR Guidance document.pdf") as pdf:
    text = "\n".join(page.extract_text() or "" for page in pdf.pages)

with open("eu_mdr_text.txt", "w", encoding="utf-8") as f:
    f.write(text)

print(f"Extracted {len(pdf.pages)} pages into 'eu_mdr_text.txt'.")
