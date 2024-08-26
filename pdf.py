from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def show_pdfs():
    # Manually specifying the PDF files
    pdf_files = ['maharashtra.pdf', 'haryana.pdf', 'gujrat.pdf', 'tamilnadu.pdf']
    return render_template('pdf.html', pdf_files=pdf_files)

@app.route('/view-pdf', methods=['GET'])
def view_pdf():
    pdf_file = request.args.get('pdf')
    return render_template('view_pdf.html', pdf_file=pdf_file)

if __name__ == '__main__':
    app.run(debug=True)
