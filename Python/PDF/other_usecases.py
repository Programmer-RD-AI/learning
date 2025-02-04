doc1 = fitz.open("file1.pdf")
doc2 = fitz.open("file2.pdf")
doc1.insert_pdf(doc2)  # Merge doc2 into doc1
doc1.save("merged.pdf")

doc = fitz.open("sample.pdf")
new_doc = fitz.open()
new_doc.insert_pdf(doc, from_page=0, to_page=1)  # Extract pages 1-2
new_doc.save("split.pdf")

page = doc[0]
page.set_rotation(90)  # Rotate 90 degrees clockwise
doc.save("rotated.pdf")
