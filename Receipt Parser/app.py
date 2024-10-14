import re
import streamlit as st
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image

# Load the Donut processor and model
processor = DonutProcessor.from_pretrained("debu-das/donut_receipt_v1.20")
model = VisionEncoderDecoderModel.from_pretrained("debu-das/donut_receipt_v1.20")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def process_document(image):
    # Prepare encoder inputs
    pixel_values = processor(image, return_tensors="pt").pixel_values

    # Prepare decoder inputs
    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

    # Generate answer
    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    # Postprocess
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # Remove first task start token

    result_json = processor.token2json(sequence)
    return result_json  # Ensure this is returning valid JSON

def summarize_text(text):
    # Prepare a human-readable string from the JSON
    invoice_no = text.get('invoice_no', 'N/A')
    invoice_total = text.get('invoice_total', 'N/A')
    vendor = text.get('vendor', 'N/A')
    line_items = text.get('line_items', [])

    # Create a formatted list of line items for the summary
    items_summary = []
    for index, item in enumerate(line_items, start=1):  # Start numbering from 1
        item_summary = f"{index}. {item['quantity']}x {item['description']} (Total: {item['total']})"
        items_summary.append(item_summary)

    # Create a clean summary statement
    line_items_str = '\n    '.join(items_summary)  # Indent for a new line format
    summary_text = (
        f"Invoice Number: {invoice_no}\n"
        f"Vendor: {vendor}\n"
        f"Total Amount: {invoice_total}\n"
        f"Items Purchased:\n    {line_items_str}\n"  # Indent and add line breaks
    )

    return summary_text  # Return the formatted summary text directly

# Streamlit app
st.title("Receipt Parsing")
st.markdown(""" 
    This app uses Donut, an instance of `VisionEncoderDecoderModel` fine-tuned on CORD (document parsing).
    To use it, simply upload your image and click 'Submit'.
""")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    if st.button('Submit'):
        with st.spinner('Processing...'):
            result = process_document(image)
            st.json(result)  # Display the JSON result
            
            # Summarize the result
            text_to_summarize = result  # Use the JSON directly for summarization

            # Display the formatted summary with improved spacing
            st.subheader("Summary:")
            st.markdown(f"* **Invoice Number:** {text_to_summarize.get('invoice_no', 'N/A')}")
            st.markdown(f"* **Vendor:** {text_to_summarize.get('vendor', 'N/A')}")
            st.markdown(f"* **Total Amount:** {text_to_summarize.get('invoice_total', 'N/A')}")
            st.markdown(f"**Items Purchased:**")

            # Create a list of line items
            line_items = text_to_summarize.get('line_items', [])
            for index, item in enumerate(line_items, start=1):
                st.write(f"{index}. {item['quantity']}x {item['description']} (Total: {item['total']})")

            
