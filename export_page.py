import pymupdf
doc = pymupdf.open("knowledge_docs\\RMS_USER_MANUALv7.pdf")
page = doc[43]  # 44页，pymupdf从0开始数，所以填43
pix = page.get_pixmap(dpi=200)
pix.save("eop_flow.png")
print("saved eop_flow.png")