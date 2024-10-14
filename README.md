# Receipt Parsing App with Donut

This project is a Streamlit web application that uses the Donut model (an instance of `VisionEncoderDecoderModel`) fine-tuned on CORD for receipt parsing. The app takes an image of a receipt, extracts structured data such as invoice number, vendor, total amount, and line items, and provides a formatted summary of the receipt contents.

## Features

- **Upload an image**: Users can upload receipt images in `.png`, `.jpg`, or `.jpeg` format.
- **Text extraction**: The app processes the image and extracts relevant details like the invoice number, vendor, and total amount.
- **Formatted summary**: The extracted data is displayed in a human-readable format.

## Setup Instructions

### Prerequisites

- Python 3.12.7 (or any Python 3.12 version)
- `pip` for installing dependencies

### Installation

1. Clone the repository:
    ```bash
    [git clone https://github.com/your-username/receipt-parsing-app.git](https://github.com/kazumanjay07/ReceiptParser.git)
    cd receipt-parsing-app
    ```

2. Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

### Usage

1. Open the web browser at the URL displayed in the terminal (default: `http://localhost:8501`).
2. Upload a receipt image in `.png`, `.jpg`, or `.jpeg` format.
3. Click the "Submit" button to process the image.
4. View the extracted data and summary of the receipt.

### Model

The model used in this app is the `debu-das/donut_receipt_v1.20`, a fine-tuned version of the Donut model for document parsing.

### Project Structure

- `app.py`: The main script for running the Streamlit application.
- `requirements.txt`: Contains the Python dependencies needed to run the app.

### Example

1. **Upload a receipt image**: Upload any receipt image file in `.png`, `.jpg`, or `.jpeg` format.
2. **View extracted data**: The app will display the raw JSON data extracted from the receipt.
3. **View summary**: A formatted summary of the receipt will be displayed below the raw JSON data.

## Dependencies

Please check the `requirements.txt` file for the list of Python dependencies.

## License

This project is licensed under the MIT License.
