import json
from distutils.dir_util import copy_tree
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
        <title>OCR Comparison</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
            th, td { border: 1px solid #dddddd; text-align: left; padding: 8px; }
            th { background-color: #f2f2f2; }
            img { max-width: 200px; margin: 5px; }
        </style>
    </head>
    <body>
        <h1>Text Recognition Benchmark</h1>
        <table>
            <tr>
                <th>Tool Name</th>
                <th>Average CER</th>
                <th>Average WER</th>
                <th>Average Processing Time (s)</th>
                <th>More</th>
            </tr>
    """
    sorted_tools = dict(sorted(metrics.items(), key=lambda item: item[1]['avg_cer_text'], reverse=False))

    for tool_name, data in sorted_tools.items():
        html_content += f"""
            <tr>
                <td>{tool_name}</td>
                <td>{data['avg_cer_text']:.2f}</td>
                <td>{data['avg_wer_text']:.2f}</td>
                <td>{data['avg_processing_time']:.2f}</td>
                <td><a href="./{tool_name}-metrics.html">More...</a></td>
            </tr>
        """
    html_content += """
        </table>
        <br>
        <h1>Number Recognition Benchmark</h1>
        <table>
            <tr>
                <th>Tool Name</th>
                <th>Average CER</th>
                <th>Average WER</th>
                <th>Average Processing Time (s)</th>
                <th>More</th>
            </tr>
    """
    sorted_tools = dict(sorted(metrics.items(), key=lambda item: item[1]['avg_cer_number'], reverse=False))
    for tool_name, data in sorted_tools.items():
        html_content += f"""
            <tr>
                <td>{tool_name}</td>
                <td>{data['avg_cer_number']:.2f}</td>
                <td>{data['avg_wer_number']:.2f}</td>
                <td>{data['avg_processing_time']:.2f}</td>
                <td><a href="./{tool_name}-metrics.html">More...</a></td>
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
            <h1>{tool_name} | Text Recognition</h1>
            <table>
                <tr>
                    <th>Image</th>
                    <th>Expected</th>
                    <th>Recognized</th>
                    <th>CER</th>
                    <th>WER</th>
                    <th>Processing Time (s)</th>
                </tr>
        """

        sorted_tools = dict(sorted([(image_id, data) for image_id, data in metrics[tool_name]["images"].items() if not image_id.startswith('t')], key=lambda item: item[1]['cer'], reverse=False))

        for image_id, data in sorted_tools.items():
            html_content += f"""
                <tr>
                    <td><img src="images/{image_id}.png" height="32"></td>
                    <td>{data['text']}</td>
                    <td>{data['recognized']}</td>
                    <td>{data['cer']:.2f}</td>
                    <td>{data['wer']:.2f}</td>
                    <td>{data['processing_time']:.2f}</td>
                </tr>
            """
        html_content += f"""
            </table>
            <br>
            <h1>{tool_name} | Number Recognition</h1>
            <table>
                <tr>
                    <th>Image</th>
                    <th>Expected</th>
                    <th>Recognized</th>
                    <th>CER</th>
                    <th>WER</th>
                    <th>Processing Time (s)</th>
                </tr>
        """

        sorted_tools = dict(sorted([(image_id, data) for image_id, data in metrics[tool_name]["images"].items() if not image_id.startswith('n')], key=lambda item: item[1]['cer'], reverse=False))

        for image_id, data in sorted_tools.items():
            html_content += f"""
                <tr>
                    <td><img src="images/{image_id}.png" height="32"></td>
                    <td>{data['text']}</td>
                    <td>{data['recognized']}</td>
                    <td>{data['cer']:.2f}</td>
                    <td>{data['wer']:.2f}</td>
                    <td>{data['processing_time']:.2f}</td>
                </tr>
            """

        html_content += """
            </table>
        </body>
        </html>
        """

        with(open('./recognition-benchmark/output/' + tool_name + '-metrics.html', 'w') as f):
            f.write(html_content)


def save_html(html, output_filename):
    with open(output_filename, 'w') as f:
        f.write(html)

def main():
    metrics_file = 'recognition-benchmark/output/metrics.json'
    output_html_file = 'recognition-benchmark/output/metrics_comparison.html'

    metrics = load_metrics(metrics_file)
    html = generate_html(metrics)
    generate_html_images(metrics)
    save_html(html, output_html_file)
    copy_tree("recognition-benchmark/dataset/images", "recognition-benchmark/output/images")
    print(f"HTML file generated: {output_html_file}")

if __name__ == "__main__":
    main()

