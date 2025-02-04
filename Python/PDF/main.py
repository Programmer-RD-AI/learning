import fitz  # PyMuPDF

doc = fitz.open("sample.pdf")  # Open a PDF
print("Number of pages:", len(doc))
page = doc[0]  # First page (zero-indexed)
print("Page size:", page.rect)  # Dimensions of the page
text = page.get_text("text")  # Extract text
print(text)
search_result = page.search_for("keyword")
print("Keyword found at:", search_result)  # Coordinates of the matches
for img_index, img in enumerate(page.get_images(full=True)):
    xref = img[0]
    base_image = doc.extract_image(xref)
    image_bytes = base_image["image"]
    image_ext = base_image["ext"]
    with open(f"image_{img_index}.{image_ext}", "wb") as f:
        f.write(image_bytes)
rect = fitz.Rect(50, 50, 200, 100)  # Define a rectangle
annot = page.add_highlight_annot(rect)
page.insert_textbox(rect, "Highlighted text")
doc.save("annotated.pdf")
print("Metadata:", doc.metadata)
doc.set_metadata({"title": "New Title"})
doc.save("updated.pdf")
doc.save("output.pdf")
