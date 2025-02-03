import fitz  # PyMuPDF

# Open the PDF file
pdf_document = "output.pdf"
doc = fitz.open(pdf_document)

# Iterate over pages to extract text and images
for page_number in range(len(doc)):
    page = doc[page_number]
    text = page.get_text("text")
    print(f"--- Text on page {page_number + 1} ---")
    print(text)

    # Extract images
    image_list = page.get_images(full=True)
    for image_index, img in enumerate(image_list, start=1):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]
        image_filename = f"page{page_number + 1}_img{image_index}.{image_ext}"
        with open(image_filename, "wb") as image_file:
            image_file.write(image_bytes)
        print(f"Extracted image saved as {image_filename}")

    # Extract embedded files (could include audio, if any)
    for emb in doc.embeddedFileInfos():
        print("Found embedded file:", emb["name"])
        file_data = doc.extract_embedded_file(emb["xref"])
        with open(emb["name"], "wb") as f:
            f.write(file_data)
doc.close()
