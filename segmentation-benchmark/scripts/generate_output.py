import json
import os

def load_metrics(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def generate_html(metrics):
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Text Detectors Comparison</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
            th, td { border: 1px solid #dddddd; text-align: left; padding: 8px; }
            th { background-color: #f2f2f2; }
            img { max-width: 200px; margin: 5px; }
        </style>
    </head>
    <body>
        <h1>Text Segmentation Comparison</h1>
        <table>
            <tr>
                <th>Tool Name</th>
                <th>Average Processing Time (s)</th>
                <th>Average IOU</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1 Score</th>
                <th>More</th>
            </tr>
    """

    sorted_tools = dict(sorted(metrics.items(), key=lambda item: item[1]['avg_f1_score'], reverse=True))

    for tool_name, data in sorted_tools.items():
        html_content += f"""
            <tr>
                <td>{tool_name}</td>
                <td>{data['avg_processing_time']:.2f}</td>
                <td>{data['avg_iou']:.2f}</td>
                <td>{data['avg_precision']:.2f}</td>
                <td>{data['avg_recall']:.2f}</td>
                <td>{data['avg_f1_score']:.2f}</td>
                <td><a href="./{tool_name}/metrics.html">More...</a></td>
            </tr>
        """
    html_content += """
        </table>
    </body>
    </html>
    """
    
    return html_content


def generate_html_images(metrics):
    for tool_name, data in metrics.items():
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Text Detectors Comparison</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
                th, td { border: 1px solid #dddddd; text-align: left; padding: 8px; }
                th { background-color: #f2f2f2; }
                img { max-width: 200px; margin: 5px; }
            </style>
        </head>
        """
        html_content += f"""
        <body>
            <h1>{tool_name} Text Detectors Comparison</h1>
            <table>
                <tr>
                    <th>Image id</th>
                    <th>Processing Time (s)</th>
                    <th>IOU</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1 Score</th>
                    <th>Image</th>
                </tr>
        """

        for image_id, data in metrics[tool_name]["images"].items():
            html_content += f"""
                <tr>
                    <td>{image_id}</td>
                    <td>{data['processing_time']:.2f}</td>
                    <td>{data['iou_avg']:.2f}</td>
                    <td>{data['precision']:.2f}</td>
                    <td>{data['recall']:.2f}</td>
                    <td>{data['f1_score']:.2f}</td>
                    <td><a href="{image_id}.png">View Image</a></td>
                </tr>
            """
        html_content += """
            </table>
        </body>
        </html>
        """

        with(open('./segmentation-benchmark/output/' + tool_name + '/metrics.html', 'w') as f):
            f.write(html_content)
    
    # return html_content


    # <tr>
    #     <td colspan="6">
    #         <h3>Images:</h3>

# for image_id, image_data in data['images'].items():
#     html_content += f"""
#         <a href="#img_{tool_name}_{image_id}">
#             Image {image_id}
#         </a>
#         <div id="img_{tool_name}_{image_id}">
#             <img src="{image_id}.jpg" alt="Image {image_id} from {tool_name}">
#             <div>
#                 <strong>Processing Time:</strong> {image_data['processing_time']:.2f}s<br>
#                 <strong>IOU Average:</strong> {image_data['iou_avg']:.2f}<br>
#                 <strong>Precision:</strong> {image_data['precision']:.2f}<br>
#                 <strong>Recall:</strong> {image_data['recall']:.2f}<br>
#                 <strong>F1 Score:</strong> {image_data['f1_score']:.2f}<br>
#             </div>
#         </div>
#     """

# html_content += "</td></tr>"


def save_html(html, output_filename):
    with open(output_filename, 'w') as f:
        f.write(html)

def main():
    metrics_file = 'segmentation-benchmark/output/metrics.json'
    output_html_file = 'segmentation-benchmark/output/metrics_comparison.html'
    
    metrics = load_metrics(metrics_file)    
    html = generate_html(metrics)
    generate_html_images(metrics)
    save_html(html, output_html_file)
    print(f"HTML file generated: {output_html_file}")

if __name__ == "__main__":
    main()

# # Write script to load metrics.json file and generate html file based on the data (simple table) metrics has this structure:

# there are multiple files. create table to compare different text detectors.
# it should have these columns (tool name/image, processing time, iou average, precision, recall, f1 score) below of each tool show images and metrics, then next tool with it's image....

# { "tesseract": { "avg_processing_time": 20.749979066848756, "avg_iou": 0.2301887434617711, "avg_precision": 1.0, "avg_recall": 0.1881091105349086, "avg_f1_score": 0.2984240652639676, "images": {
#       "1": {
#         "processing_time": 8.909185886383057,
#         "iou_scores": [],
#         "iou_avg": 0.28235506492162277,
#         "precision": 1.0,
#         "recall": 0.16666666666666666,
#         "f1_score": 0.2857142857142857
#       },
#     ...
#     }
#   },
#   "another_tesseract": {
#     "avg_processing_time": 15.275244736671448,
#     "avg_iou": 0.2301887434617711,
#     "avg_precision": 1.0,
#     "avg_recall": 0.1881091105349086,
#     "avg_f1_score": 0.2984240652639676,
#     "images": {
#       "1": {
#         "processing_time": 11.773192405700684,
#         "iou_scores": [],
#         "iou_avg": 0.28235506492162277,
#         "precision": 1.0,
#         " recall": 0.16666666666666666,
#         "f1_score": 0.2857142857142857
#       },
#         ...
#     }
#   }
# }

