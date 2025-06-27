from flask import Flask, render_template, request, url_for
import plotly.express as px
import pandas as pd
from vogue2 import fashionTrendClusterer
import numpy
import os
from PIL import Image
import numpy as np
import gc

# Initialize the Flask app
app = Flask(__name__)
topTenCluster = fashionTrendClusterer()
topTenCluster.load_dataset()
topTenCluster.cluster_embeddings()  # this sets df and top_clusters
top_clusters_df = topTenCluster.get_trend_dataframe()
topTenCluster.find_representative_images()


output_dir = os.path.join("static", "images", "cluster_samples")
os.makedirs(output_dir, exist_ok=True)

#--------------------------- FOR THE TOP TRENDS PAGE.----------------------------------
""" 
    PRINTS THE TREND NAME AND DESCRIPTION FOR EACH TREND. 
    BESIDE THIS SHOULD BE TWO IMAGES CLOSEST TO ITS CLUSTER CENTROID
    """
trend_info = {
    10597: {
        "title": "Opulent Fur",
        "description": """Glamorous Fur Coats, exudes luxury and elegance.  The textures are thick, plush fur, creating a visually rich and tactile appearance.
                          The fabric is fur, which is the prominent material, exuding opulence. The silhouette is generally oversized and enveloping, emphasizing volume. The coats have a classic, sophisticated style, appealing to a high-fashion audience.
     
                       """
    },
    13641: {
        "title": "Neo-Noir Tailoring",
        "description": """A professional or academic aesthetic with a dark edge. There is a Dark Decadence defined by structured suit silhouettes, in  dark luxurious fabrics like velvet and silk, accented with delicate, vintage-inspired details.
                          With the  blazer, trousers, and footwear, it blends menswear-inspired tailoring with contemporary fashion trends contribute to a polished look."""
    },
    13893: {
        "title": "Relaxed Suits",
        "description": """Relaxed suits redefine traditional tailoring by blending structured elegance with comfortable, easygoing silhouettes. Characterized by looser fits, soft draping, and lightweight fabrics, this trend moves away from rigid, form-fitting suits in favor of a more effortless, contemporary aesthetic."""
    },
    14931: {
        "title": "Masculine Power Suits",
        "description":"""
                        Modern Tailoring with ’80s Influence    
                        The Macline Power Suit trend embodies sleek, structured tailoring with a refined yet commanding presence. Defined by sharp silhouettes, this trend features slim, straight-leg trousers paired with fitted or slightly relaxed blazers, creating a streamlined and elongated look. Inspired by 1980s power dressing, elements like padded shoulders and strong lapels add a bold, authoritative edge while maintaining a minimalist aesthetic.
                        Fabric choices lean toward smooth, high-quality weaves with a subtle sheen, ensuring a polished and sophisticated finish. The lack of visible texture or embellishment reinforces the clean, modern appeal, allowing the timeless black color to enhance the suit’s versatility and elegance. This trend balances classic tailoring with contemporary refinement, making it a staple for powerful, confident dressing in both professional and high-fashion settings
                        """
    },
    17705: {
        "title": "A-line Colourful Gowns",
        "description": """These A-line silhouette create a voluminous shape. The skirts flare dramatically starting from the waist, emphasizing the lightness of the fabric. Vibrant, colorful A-line gowns that make a bold, beautiful statement."
                          These gowns utilize sheer, lightweight fabrics, normally tulle or organza. The textures are delicate and airy, creating a sense of ethereal lightness. Implementing gradient effects, suggesting layers of varying sheerness or color saturation in the fabric and embellishments that add a subtle textural contrast.
                          A-line silhouette, creating voluminous shape. The skirts flare dramatically starting from the waist, emphasizing the lightness of the fabric.
                       """
    },
    17719: {
        "title": "Avant-Garde Romantic Ballgowns",
        "description": "The Avant-Garde Romantic Ballgown trend merges the grandeur of traditional ballgowns with experimental, boundary-pushing design elements. Characterized by voluminous silhouettes, these gowns often feature layers of tulle, silk, and lace, creating an ethereal yet commanding presence. While rooted in romantic aesthetics—seen in delicate ruching, embroidery, and soft, flowing fabrics—this trend embraces avant-garde elements through unexpected asymmetry, bold color contrasts, and unconventional textures. The color palette ranges from dark neutrals like black and burgundy, evoking a gothic elegance, to jewel tones. It is a trend that focuses on A-line or bell silhouette, often with layers of tulle or other delicate fabrics creating volume.  These frequently feature intricate detailing like ruching, embroidery, or delicate beading."
    },
    17806: {
        "title": "Feminine High fashion",
        "description": """There is a vintage romantic ode in this trend
                          Flowing silhouettes: full skirts that create a romantic and ethereal look. Typically lace, or tulle contribute to the overall flowing and feminine aesthetic.
                          Pastel color palette: The color schemes are soft and delicate, enhancing the romantic feel.
                          Intricate detailing: sequins and the layering of tulle is used to create texture and interest. But also other features like intricate embroidery and lacework are present in this thematic trend.
                          """
    },
    19243: {
        "title": "Tulle Dresses",
        "description": """These styles combine flowing, feminine silhouettes with dark, moody elements. Key aspects include: Maxi or midi-length dresses, often with sheer or layered tulle fabrics; Use of black as the primary color, conveying a sense of mystery and drama;
                          Delicate details like floral embellishments, lace, or corsetry-inspired bodices, adding a touch of romance; A balance between soft, ethereal textures and more structured, edgy elements, creating a sophisticated gothic aesthetic.
                          The skirt of the gown on the right, like its counterpart, is layered tulle, contributing to a sense of movement and airiness. There is a common use of layered tulle to create a romantic effect."""

    },
    19418: {
        "title": "Off the Shoulder Maxi Dresses",
        "description": """Off-the-shoulder maxi dresses embody graceful sophistication, blending romantic allure with a relaxed, flowing silhouette. Defined by their bare-shoulder neckline, these dresses highlight the collarbone and shoulders, creating a subtly sensual yet refined look. The maxi length adds an element of drama and elegance, making them perfect for both casual and formal occasions.
                         Fabric choices range from lightweight chiffon and breezy cotton for a soft, ethereal feel to silk and satin for a more luxurious, evening-ready aesthetic. Some designs feature ruffled or draped necklines, enhancing the romantic appeal, while others opt for sleek, minimalist cuts for a modern touch. The color palette varies from soft pastels and floral prints for a feminine, bohemian vibe to rich jewel tones and monochromatic hues for a bold, sophisticated statement.
                        Off-the-shoulder maxi dresses offer a timeless blend of effortless beauty and graceful movement, making them a staple in contemporary fashion."""

    },
    31040: {
        "title": "Sleek Black Dresses",
        "description": """Sleek Black  dresses are a staple. These will never go out of style. They showcase elongated, columnar silhouettes. 
                          They are very streamlined, emphasizing a long, lean, and vertical line. There's a lack of significant volume or shape-defining elements. 
                          Some more elegant black dresses utilise draping and a slight, subtle cascade of fabric at the hem. They can be fun and have slight asymmetry in the drape, or just a simple dress that is strictly rectangular.
                          They are overall sleek and sophisticated"""
    }
}

#Retrieve images clsotest to cluster centroid

clusters = []
for cluster_id in topTenCluster.top_clusters:
    rep_indices = topTenCluster.rep_indices.get(cluster_id, [])[:2]
    image_paths = []
    
    # Fetch the trend title and description from the dictionary using the cluster_id
    trend_data = trend_info.get(cluster_id, {"title": "Unknown Trend", "description": "Description not available"})
    
    # Set the title and description for each trend
    title = trend_data["title"]
    description = trend_data["description"]

    for i, idx in enumerate(rep_indices):
        image_data = topTenCluster.dataset[idx]['image']

        # Handle image extraction
        if isinstance(image_data, np.ndarray):
            img = Image.fromarray(image_data)
        else:
            img = image_data  #PIL image

        filename = f"cluster{cluster_id}_img{i}.jpg"
        filepath = os.path.join(output_dir, filename) 
        if not os.path.exists(filepath):  # prevents duplicate writing
            img.convert("RGB").save(filepath, format='JPEG', optimize=True)

        image_paths.append(f'images/cluster_samples/{filename}')
        
        del img  # free memory

    # Append the trend data to the clusters
    clusters.append({
        "cluster_id": cluster_id,
        "images": image_paths,
        "title": title,  # Uses the title from the dictionary above
        "description": description  # Uses the description from the dictionary above
    })
    gc.collect()  # free memory aggressively

    

# Route for the homepage
@app.route('/')
def home():
    return render_template("home.html")

@app.route('/trend-forecaster')
def TrendForecast():
    forecast_htmls = {}
    html_dir = "static/forecast"
    for filename in os.listdir(html_dir):
        if filename.endswith(".html"):
            cluster_id = filename.split("_")[1].split(".")[0]
            with open(os.path.join(html_dir, filename), "r") as f:
                forecast_htmls[cluster_id] = f.read()
    return render_template("TrendForecast.html",forecast_htmls=forecast_htmls)

@app.route('/top-trends')
def TopTrends():
    cluster_html = topTenCluster.visualize_clusters()
    return render_template("TopTrends.html", cluster_graph=cluster_html, clusters=clusters)

@app.route('/designer-influence')
def DesignerInfluence():
    return render_template("DesignerInfluence.html")

@app.route('/about')
def about():
    return render_template("about.html")


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
