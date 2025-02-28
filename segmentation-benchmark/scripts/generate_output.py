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
        <h1>Comparison of Text Segmentation Tools</h1>
        <p>Precision, Recall, and F1 Score – Higher numbers mean better performance. The best tools are listed at the top.</p>
        <table>
            <tr>
                <th>Tool Name</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1 Score</th>
                <th>Average Processing Time</th>
                <th>More</th>
            </tr>
    """

    sorted_tools = dict(sorted(metrics.items(), key=lambda item: item[1]['avg_f1_score'], reverse=True))

    for tool_name, data in sorted_tools.items():
        html_content += f"""
            <tr style="opacity: {int(data['avg_f1_score'] * 100)}%;">
                <td>{tool_name}</td>
                <td>{int(data['avg_precision'] * 100)}%</td>
                <td>{int(data['avg_recall'] * 100)}%</td>
                <td>{int(data['avg_f1_score'] * 100)}%</td>
                <td>{int(data['avg_processing_time'])}s</td>
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
                th, td { border: 1px solid #bbbbbb; text-align: left; padding: 8px; }
                tbody tr td {background: transparent; }
                th { background-color: #f2f2f2; }
                img { max-width: 200px; margin: 5px; }
            </style>
        </head>
        """
        html_content += f"""
        <body>
            <h1>{tool_name} | Text Detector</h1>
            <table>
                <tr>
                    <th>Image id</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1 Score</th>
                    <th>Processing Time</th>
                    <th>Image</th>
                </tr>
        """

        for image_id, data in metrics[tool_name]["images"].items():
            html_content += f"""
                <tr style="background-color: rgba({(1 - data['f1_score'])*255}, {data['f1_score']*255}, 0, 0.4)">
                    <td>{image_id}</td>
                    <td>{int(data['precision'] * 100)}%</td>
                    <td>{int(data['recall']*100)}%</td>
                    <td>{int(data['f1_score']* 100)}%</td>
                    <td>{int(data['processing_time'])}s</td>
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

