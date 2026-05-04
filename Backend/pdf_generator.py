import os
import io
import requests
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

STRATEGIC_RECOMMENDATIONS = {
    "FMCG": [
        "Maximize Sensory Cues: Highlight the sounds and textures of the product. These 'sensory triggers' create an immediate craving and emotional connection faster than words.",
        "Focus on Functional Benefits: Don't just list ingredients; explain their value. Clear, benefit-driven communication builds transparency and removes consumer hesitation.",
        "Sell Experience Over Price: Justify the price by focusing on 'Lifestyle Benefits' like saving time or improving mood. Consumers pay more when they see a solution to a personal need.",
        "Reduce Cognitive Clutter: Use minimalist designs and simple messaging. Reducing the 'mental effort' required to understand the brand makes the product feel more premium and easier to buy.",
        "Show Responsive Evolution: Publicly acknowledge that product updates are based on customer feedback. This 'We Listened' approach builds trust and turns critics into loyal advocates."
    ],
    "Beauty & Fragrances": [
        "Synchronize Senses with Visuals: Align the product’s texture and scent with the advertisement’s 'vibe' to eliminate consumer doubt and build immediate subconscious trust.",
        "Prioritize Precision Ergonomics: Optimize applicators (brushes, pumps, or tips) for effortless use, as a smooth 'first-contact' experience directly increases perceived product quality.",
        "Demonstrate Instant Value for Pricing: Highlight immediate, visible results (e.g., '8h wear' or 'instant volume') to emotionally justify premium price points and reduce 'price pain'.",
        "Focus on Emotional Efficacy: Translate technical formulas into lifestyle benefits (e.g., 'flexible feel' instead of 'chemical names') to create a stronger emotional bond with the user.",
        "Master 'Honest Luxury': Ensure the advertisement’s promises match the sample’s performance to avoid 'expectation shock' and secure long-term brand loyalty."
    ],
    "Sportswear & Athletics": [
        "Synchronize Kinetic Energy: Ensure the product’s physical weight and flexibility perfectly mirror the high-speed movement shown in the ad to prevent 'Expectation Shock' during the first trial.",
        "Materialize Technical Promises: Focus on making cushioning and breathability technologies immediately 'felt' by the user to validate the premium pricing through tangible performance.",
        "Optimize the 'First-Minute' Experience: Refine the ease of wear and initial comfort, as the brain decides on the product’s 'Professional Grade' within the first 60 seconds of use.",
        "Anchor Price to Sustainability & Durability: Highlight eco-friendly materials and long-term resilience in both the ad and packaging to shift the consumer’s mindset from 'High Cost' to 'Reliable Investment'.",
        "Eliminate Sensory Dissonance: Align the 'Visual Power' of the advertisement with the 'Tactile Quality' of the product; if the ad looks powerful but the sample feels flimsy, brand loyalty drops instantly."
    ],
    "Consumer Electronics": [
        "Validate 'Smart Claims' through Immediate Utility: Ensure that any advanced software or AI features deliver a visible result within the first 30 seconds of interaction.",
        "Align Tactical 'Feel' with Hardware Reliability: Use premium materials that provide a sense of physical 'Solidarity'. The brain subconsciously links the weight and texture to durability.",
        "Prioritize 'Zero-Latency' Sensory Response: Maximize the speed of the interface and physical controls. Any delay creates immediate cognitive friction.",
        "Translate Specifications into Functional Benefits: Shift the marketing and demo focus from raw numbers (Specs) to 'Lifestyle Wins'. Demonstrate how the technology solves a daily problem.",
        "Minimize Cognitive Load through Intuitive Design: Streamline the user interface to ensure that advanced features do not create 'Tech-Anxiety'. A high-end device should feel powerful yet effortless."
    ]
}

def generate_analytics_pdf(company_name, company_logo, product_name, product_image, product_description, stats, category):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], alignment=TA_CENTER, fontSize=24, textColor=colors.HexColor("#0f172a"), spaceAfter=10)
    subtitle_style = ParagraphStyle('SubtitleStyle', parent=styles['Normal'], alignment=TA_CENTER, fontSize=12, textColor=colors.HexColor("#64748b"), spaceAfter=20)
    heading_style = ParagraphStyle('HeadingStyle', parent=styles['Heading2'], fontSize=16, textColor=colors.HexColor("#1e293b"), spaceAfter=10, spaceBefore=20)
    normal_style = ParagraphStyle('NormalStyle', parent=styles['Normal'], fontSize=10, textColor=colors.HexColor("#334155"), leading=14, spaceAfter=10)
    bullet_style = ParagraphStyle('BulletStyle', parent=styles['Normal'], fontSize=11, textColor=colors.HexColor("#334155"), leading=16, spaceAfter=8, bulletIndent=10, leftIndent=25)
    metric_style = ParagraphStyle('MetricStyle', parent=styles['Normal'], alignment=TA_CENTER)
    
    # --- PAGE 1: Overview & Metrics ---
    
    # Header Table for Logo and Title
    header_data = []
    logo_img = None
    if company_logo:
        try:
            if company_logo.startswith("http"):
                response = requests.get(company_logo)
                img_data = io.BytesIO(response.content)
                logo_img = RLImage(img_data, width=1.5*inch, height=1.5*inch)
            else:
                logo_path = os.path.join(os.getcwd(), company_logo.lstrip("/"))
                if os.path.exists(logo_path):
                    logo_img = RLImage(logo_path, width=1.5*inch, height=1.5*inch)
        except Exception as e:
            print("Error loading company logo:", e)
    
    if logo_img:
        header_data.append([logo_img, Paragraph(f"<b>{company_name}</b><br/>Neuromarketing Analytics Report", title_style)])
    else:
        header_data.append(["", Paragraph(f"<b>{company_name}</b><br/>Neuromarketing Analytics Report", title_style)])
        
    header_table = Table(header_data, colWidths=[2*inch, 4.5*inch])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (1,0), (1,0), 'CENTER'),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 20))
    
    # Product Details
    story.append(Paragraph("Advertised Product Profile", heading_style))
    
    product_data = []
    prod_img = None
    if product_image:
        try:
            if product_image.startswith("http"):
                response = requests.get(product_image)
                img_data = io.BytesIO(response.content)
                prod_img = RLImage(img_data, width=2.5*inch, height=2.5*inch)
            else:
                prod_path = os.path.join(os.getcwd(), product_image.lstrip("/"))
                if os.path.exists(prod_path):
                    prod_img = RLImage(prod_path, width=2.5*inch, height=2.5*inch)
        except Exception as e:
            print("Error loading product image:", e)
            
    desc_para = Paragraph(product_description or "No description provided.", normal_style)
    name_para = Paragraph(f"<b>{product_name}</b>", ParagraphStyle('ProdName', parent=styles['Heading3'], fontSize=14, textColor=colors.HexColor("#0f172a")))
    
    if prod_img:
        product_data.append([prod_img, [name_para, Spacer(1, 10), desc_para]])
        prod_table = Table(product_data, colWidths=[3*inch, 3.5*inch])
    else:
        product_data.append([[name_para, Spacer(1, 10), desc_para]])
        prod_table = Table(product_data, colWidths=[6.5*inch])
        
    prod_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('PADDING', (0,0), (-1,-1), 15),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#e2e8f0")),
    ]))
    story.append(prod_table)
    story.append(Spacer(1, 30))
    
    # KPI Metrics
    story.append(Paragraph("Key Performance Metrics", heading_style))
    
    total_res = stats.get("total_respondents", 0)
    pos = stats.get("overall_positivity", 0)
    neg = stats.get("resistance_index", 0)
    res_score = stats.get("ad_resonance_score", 0)
    
    metrics_data = [
        [
            [Paragraph("<font size=11 color='#64748b'>Total Respondents</font>", metric_style),
             Spacer(1, 8),
             Paragraph(f"<font size=24 color='#0f172a'><b>{total_res}</b></font>", metric_style)],
            
            [Paragraph("<font size=11 color='#64748b'>Overall Positivity</font>", metric_style),
             Spacer(1, 8),
             Paragraph(f"<font size=24 color='#10b981'><b>{pos}%</b></font>", metric_style)]
        ],
        [
            [Paragraph("<font size=11 color='#64748b'>Resistance Index</font>", metric_style),
             Spacer(1, 8),
             Paragraph(f"<font size=24 color='#f43f5e'><b>{neg}%</b></font>", metric_style)],
             
            [Paragraph("<font size=11 color='#64748b'>Ad Resonance Score</font>", metric_style),
             Spacer(1, 8),
             Paragraph(f"<font size=24 color='#0f172a'><b>{res_score}/100</b></font>", metric_style)]
        ]
    ]
    
    metrics_table = Table(metrics_data, colWidths=[3.25*inch, 3.25*inch], rowHeights=[1*inch, 1*inch])
    metrics_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#e2e8f0")),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f1f5f9")),
        ('PADDING', (0,0), (-1,-1), 15),
    ]))
    story.append(metrics_table)
    
    story.append(PageBreak())
    
    # --- PAGE 2: Strategic Recommendations ---
    
    # Mapping exact category names from db/frontend if they differ slightly
    cat_key = None
    for k in STRATEGIC_RECOMMENDATIONS.keys():
        if category and k.lower() in category.lower():
            cat_key = k
            break
            
    if not cat_key:
        cat_key = "FMCG" # Default fallback
        
    bullets = STRATEGIC_RECOMMENDATIONS[cat_key]
    
    # Rephrase context based on positivity
    if pos > 50:
        rec_title = "Strengths to Maintain & Scale"
        rec_context = f"The campaign for <b>{product_name}</b> achieved an impressive overall positivity of {pos}%. To maintain this momentum and further solidify consumer trust in the {cat_key} market, we recommend focusing on the following strategic pillars:"
        bullet_color = colors.HexColor("#10b981")
    else:
        rec_title = "Critical Areas for Improvement"
        rec_context = f"The campaign for <b>{product_name}</b> recorded a high resistance index of {neg}%, indicating significant cognitive friction among consumers. To improve ad resonance and conversion rates in the {cat_key} market, it is critical to implement the following strategic adjustments:"
        bullet_color = colors.HexColor("#f43f5e")

    story.append(Paragraph(f"Strategic Recommendations: {cat_key}", heading_style))
    story.append(Paragraph(rec_context, normal_style))
    story.append(Spacer(1, 15))
    
    story.append(Paragraph(f"<font color='{bullet_color}'><b>{rec_title}</b></font>", ParagraphStyle('Sub', parent=styles['Heading3'], fontSize=14, spaceAfter=15)))
    
    for bullet in bullets:
        parts = bullet.split(":", 1)
        if len(parts) == 2:
            formatted_bullet = f"<b>{parts[0]}:</b>{parts[1]}"
        else:
            formatted_bullet = bullet
            
        # Draw bullet point
        p = Paragraph(f"<bullet>&bull;</bullet>{formatted_bullet}", bullet_style)
        story.append(p)
    
    # Build PDF
    doc.build(story)
    
    buffer.seek(0)
    return buffer.getvalue()

def generate_system_pdf(stats):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], alignment=TA_CENTER, fontSize=24, textColor=colors.HexColor("#0f172a"), spaceAfter=10)
    subtitle_style = ParagraphStyle('SubtitleStyle', parent=styles['Normal'], alignment=TA_CENTER, fontSize=12, textColor=colors.HexColor("#64748b"), spaceAfter=30)
    heading_style = ParagraphStyle('HeadingStyle', parent=styles['Heading2'], fontSize=16, textColor=colors.HexColor("#1e293b"), spaceAfter=15, spaceBefore=20)
    metric_style = ParagraphStyle('MetricStyle', parent=styles['Normal'], alignment=TA_CENTER)
    
    # --- PAGE 1: Overview & Metrics ---
    story.append(Paragraph("<b>Neuromarketing Platform</b>", title_style))
    story.append(Paragraph("System-wide Analytics Overview", subtitle_style))
    
    # KPI Metrics
    story.append(Paragraph("Key Performance Metrics", heading_style))
    
    total_res = stats.get("total_respondents", 0)
    pos = stats.get("overall_positivity", 0)
    neg = stats.get("resistance_index", 0)
    res_score = stats.get("ad_resonance_score", 0)
    
    metrics_data = [
        [
            [Paragraph("<font size=11 color='#64748b'>Total Respondents</font>", metric_style),
             Spacer(1, 8),
             Paragraph(f"<font size=24 color='#0f172a'><b>{total_res}</b></font>", metric_style)],
            
            [Paragraph("<font size=11 color='#64748b'>Overall Positivity</font>", metric_style),
             Spacer(1, 8),
             Paragraph(f"<font size=24 color='#10b981'><b>{pos}%</b></font>", metric_style)]
        ],
        [
            [Paragraph("<font size=11 color='#64748b'>Resistance Index</font>", metric_style),
             Spacer(1, 8),
             Paragraph(f"<font size=24 color='#f43f5e'><b>{neg}%</b></font>", metric_style)],
             
            [Paragraph("<font size=11 color='#64748b'>Ad Resonance Score</font>", metric_style),
             Spacer(1, 8),
             Paragraph(f"<font size=24 color='#0f172a'><b>{res_score}/100</b></font>", metric_style)]
        ]
    ]
    
    metrics_table = Table(metrics_data, colWidths=[3.25*inch, 3.25*inch], rowHeights=[1*inch, 1*inch])
    metrics_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#e2e8f0")),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f1f5f9")),
        ('PADDING', (0,0), (-1,-1), 15),
    ]))
    story.append(metrics_table)
    
    story.append(Spacer(1, 30))
    story.append(Paragraph("Platform Engagement Summary", heading_style))
    story.append(Paragraph("The platform has aggregated data across multiple companies and campaigns. The above metrics reflect the global average and total participation across the entire ecosystem.", styles['Normal']))
    
    story.append(PageBreak())
    
    # --- PAGE 2: Company Ranking Table ---
    story.append(Paragraph("System-wide Company Engagement", heading_style))
    
    companies_data = stats.get("campaign_engagement", [])
    # Sort by engagement (positivity) descending
    companies_data.sort(key=lambda x: x.get("engagement", 0), reverse=True)
    
    table_data = [
        [
            Paragraph("<b>Company Name</b>", styles['Normal']),
            Paragraph("<b>Respondents</b>", styles['Normal']),
            Paragraph("<b>Positivity</b>", styles['Normal']),
            Paragraph("<b>Resistance</b>", styles['Normal'])
        ]
    ]
    
    for comp in companies_data:
        c_name = comp.get("name", "Unknown")
        c_res = comp.get("respondents", 0)
        c_pos = comp.get("engagement", 0)
        c_neg = 100 - c_pos if c_pos > 0 or c_res > 0 else 0
        
        table_data.append([
            Paragraph(str(c_name), styles['Normal']),
            Paragraph(str(c_res), styles['Normal']),
            Paragraph(f"<font color='#10b981'><b>{c_pos}%</b></font>", styles['Normal']),
            Paragraph(f"<font color='#f43f5e'><b>{c_neg}%</b></font>", styles['Normal'])
        ])
        
    if len(table_data) > 1:
        comp_table = Table(table_data, colWidths=[3*inch, 1.5*inch, 1.25*inch, 1.25*inch])
        comp_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#e2e8f0")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor("#0f172a")),
            ('ALIGN', (1,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
            ('PADDING', (0,0), (-1,-1), 8),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")])
        ]))
        story.append(comp_table)
    else:
        story.append(Paragraph("No company data available.", styles['Normal']))
    
    # Build PDF
    doc.build(story)
    
    buffer.seek(0)
    return buffer.getvalue()
