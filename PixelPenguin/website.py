# app.py
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('Desktop\coding projects\PixelPenguin\index.html')

@app.route('/process', methods=['POST', 'GET'])
def process():
    # Process data from button click here
    return 'Button clicked!'

if __name__ == '__main__':
    app.run(debug=True)