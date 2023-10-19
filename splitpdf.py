from PyPDF2 import PdfWriter, PdfReader

inputpdf = PdfReader(open("thesis.pdf", "rb"))
output = PdfWriter()

splits = {16:'Intro', 67:'SSL', 107:'SL', 122:'Outro'}

for i in range(len(inputpdf.pages)):
    output.add_page(inputpdf.pages[i])
    if i in splits:
        with open("splits/" + splits[i] + ".pdf", 'wb') as outputStream:
            output.write(outputStream)
            print(splits[i])
        
        output = PdfWriter()