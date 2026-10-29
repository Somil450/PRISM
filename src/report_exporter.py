"""
src/report_exporter.py
Investigative Support PDF Forensic Dossier Generator for PRISM.

Generates structured, professional, investigative support PDF forensic reports
including SHA-256 chain of custody, Dempster-Shafer mathematical mass tables,
high-resolution visual evidence panels, and LLM reasoning.
"""

from typing import Any, Dict, List, Optional
import io
import os
import hashlib
from datetime import datetime, timezone
from PIL import Image

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
    HRFlowable,
)


import subprocess

def _get_git_commit() -> str:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
        if status:
            return f"{commit}-dirty"
        return commit
    except Exception:
        return "unknown"


class ForensicReportExporter:
    """
    Exports a structured forensic laboratory dossier in PDF format for investigative support.
    """

    def __init__(self) -> None:
        self.styles = getSampleStyleSheet()
        self._init_custom_styles()

    def _init_custom_styles(self) -> None:
        self.title_style = ParagraphStyle(
            "DocTitle",
            parent=self.styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#0f172a"),
        )
        self.subtitle_style = ParagraphStyle(
            "DocSubtitle",
            parent=self.styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#64748b"),
        )
        self.heading_style = ParagraphStyle(
            "DocHeading",
            parent=self.styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#1e293b"),
            spaceBefore=10,
            spaceAfter=6,
        )
        self.body_style = ParagraphStyle(
            "DocBody",
            parent=self.styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#334155"),
        )
        self.mono_style = ParagraphStyle(
            "DocMono",
            parent=self.styles["Normal"],
            fontName="Courier",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#475569"),
        )
        self.verdict_style = ParagraphStyle(
            "DocVerdict",
            parent=self.styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=colors.white,
            alignment=1,  # Centered
        )

    def generate_pdf(
        self,
        filename: str,
        file_bytes: bytes,
        verdict: str,
        score: float,
        belief: float,
        plausibility: float,
        agent_masses: Dict[str, Dict[str, float]],
        llm_report_text: str,
        figure_png_bytes: Optional[bytes] = None,
        telemetry: Optional[Dict[str, Any]] = None,
        conflict_k: float = 0.0,
        epistemic_uncertainty: Optional[float] = None,
        **kwargs: Any,
    ) -> bytes:
        """
        Builds the PDF document in memory and returns raw PDF bytes.
        """
        # 1. Compute SHA-256 Hash of Evidence
        sha256_hash = hashlib.sha256(file_bytes).hexdigest()
        file_size_kb = len(file_bytes) / 1024.0
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        git_commit_raw = kwargs.get("git_commit")
        if not git_commit_raw or ".md" in str(git_commit_raw) or "/" in str(git_commit_raw) or "\\" in str(git_commit_raw):
            git_commit = _get_git_commit()
        else:
            git_commit = str(git_commit_raw).strip()
            try:
                status = subprocess.check_output(
                    ["git", "status", "--porcelain"],
                    stderr=subprocess.DEVNULL
                ).decode("ascii").strip()
                if status and not git_commit.endswith("-dirty"):
                    git_commit = f"{git_commit}-dirty"
            except Exception:
                pass

        u_calc = epistemic_uncertainty if epistemic_uncertainty is not None else max(0.0, 1.0 - (belief + (1.0 - plausibility)))

        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            pdf_buffer,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36,
        )

        story = []

        # Header Title
        story.append(Paragraph("PRISM FORENSIC INTELLIGENCE LAB", self.title_style))
        story.append(Paragraph("FORENSIC REPORT FOR INVESTIGATIVE SUPPORT", self.subtitle_style))
        story.append(Spacer(1, 8))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0f172a"), spaceAfter=10))

        # Chain of Custody Table
        custody_data = [
            [Paragraph("<b>Target Artifact:</b>", self.body_style), Paragraph(filename, self.mono_style),
             Paragraph("<b>Analysis Timestamp:</b>", self.body_style), Paragraph(timestamp, self.body_style)],
            [Paragraph("<b>SHA-256 Checksum:</b>", self.body_style), Paragraph(f"{sha256_hash[:32]}...<br/>{sha256_hash[32:]}", self.mono_style),
             Paragraph("<b>File Size:</b>", self.body_style), Paragraph(f"{file_size_kb:.1f} KB", self.body_style)],
            [Paragraph("<b>Forensic Engine:</b>", self.body_style), Paragraph(f"PRISM v2.4 (commit {git_commit})", self.body_style),
             Paragraph("<b>Report Status:</b>", self.body_style), Paragraph("Investigative Support (Non-Adjudicative)", self.body_style)],
        ]
        custody_table = Table(custody_data, colWidths=[110, 190, 95, 145])
        custody_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(custody_table)
        story.append(Spacer(1, 10))

        # Verdict Banner
        if verdict == "TAMPERED":
            v_bg = colors.HexColor("#dc2626")  # Red
            v_desc = "EVIDENCE STRONGLY INDICATES SYNTHETIC GENERATION OR REGIONAL TAMPERING"
        elif verdict == "AUTHENTIC":
            v_bg = colors.HexColor("#16a34a")  # Green
            v_desc = "NO SYSTEMIC MANIPULATION DETECTED ACROSS INDEPENDENT EVIDENCE STREAMS"
        else:
            v_bg = colors.HexColor("#d97706")  # Amber
            v_desc = "EVIDENCE MASS IS CONFLICTING OR AMBIGUOUS -- MANUAL AUDIT REQUIRED"

        verdict_data = [
            [Paragraph(f"VERDICT: {verdict}", self.verdict_style)],
            [Paragraph(f"<font size=8>{v_desc}</font>", ParagraphStyle("VDesc", parent=self.verdict_style, fontSize=8, leading=10))],
        ]
        verdict_table = Table(verdict_data, colWidths=[540])
        verdict_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), v_bg),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))
        story.append(verdict_table)
        story.append(Spacer(1, 8))

        # High Uncertainty Warning Box if applicable
        if u_calc >= 0.60 or conflict_k >= 0.50:
            warn_p = Paragraph(
                f"<b>[!] HIGH FORENSIC UNCERTAINTY NOTICE:</b> Epistemic uncertainty m(U) = {u_calc:.1%} "
                f"or conflict K = {conflict_k:.3f} is elevated. Signals across forensic modalities disagree or lack conclusive power. "
                "This assessment must be considered <b>INCONCLUSIVE</b> and cannot serve as legal determination.",
                ParagraphStyle("WarnP", parent=self.body_style, textColor=colors.HexColor("#92400e"), fontSize=8, leading=11)
            )
            warn_table = Table([[warn_p]], colWidths=[540])
            warn_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fef3c7")),
                ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#f59e0b")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]))
            story.append(warn_table)
            story.append(Spacer(1, 8))

        # Dempster-Shafer Interval Bounds Row (including Conflict K)
        math_summary = [
            [
                Paragraph(f"<b>Pignistic BetP(T):</b> {score:.1%}", self.body_style),
                Paragraph(f"<b>Belief Bel(T):</b> {belief:.1%}", self.body_style),
                Paragraph(f"<b>Plausibility Pl(T):</b> {plausibility:.1%}", self.body_style),
                Paragraph(f"<b>Uncertainty m(U):</b> {u_calc:.1%}", self.body_style),
                Paragraph(f"<b>Conflict K:</b> {conflict_k:.3f}", self.body_style),
            ]
        ]
        math_table = Table(math_summary, colWidths=[108, 108, 108, 108, 108])
        math_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(math_table)
        story.append(Spacer(1, 10))

        # Section 1: Multi-Agent Evidence Allocation Table (with Per-Agent Scores)
        story.append(Paragraph("1. Multi-Agent Forensic Scores & Evidence Allocation", self.heading_style))
        agent_scores_dict = kwargs.get("agent_scores", {})
        agent_table_data = [
            [
                Paragraph("<b>Forensic Stream</b>", self.body_style),
                Paragraph("<b>Modality / Method</b>", self.body_style),
                Paragraph("<b>P(Tampered)</b>", self.body_style),
                Paragraph("<b>m(Tampered)</b>", self.body_style),
                Paragraph("<b>m(Authentic)</b>", self.body_style),
                Paragraph("<b>m(Uncertainty)</b>", self.body_style),
            ]
        ]
        for name, m in agent_masses.items():
            clean_name = name.replace("_mass", "").replace("_", " ").title()
            short_key = name.lower().replace("_mass", "").replace("agent", "").strip()
            # Find matching score if available
            p_score = agent_scores_dict.get(name) or agent_scores_dict.get(short_key)
            if p_score is None:
                # Estimate pignistic from mass if not explicitly provided
                p_score = m.get("T", 0.0) + 0.5 * m.get("U", 0.0)
            agent_table_data.append([
                Paragraph(clean_name, self.body_style),
                Paragraph("Signal / Physics", self.body_style),
                Paragraph(f"{float(p_score):.3f}", self.mono_style),
                Paragraph(f"{m.get('T', 0.0):.3f}", self.mono_style),
                Paragraph(f"{m.get('A', 0.0):.3f}", self.mono_style),
                Paragraph(f"{m.get('U', 0.0):.3f}", self.mono_style),
            ])

        agent_table = Table(agent_table_data, colWidths=[140, 100, 75, 75, 75, 75])
        agent_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(agent_table)
        story.append(Spacer(1, 8))

        # Measured Benchmark Error Rates Section (from docs/EVAL_LOG.md Entry 003)
        story.append(Paragraph("2. Measured Pipeline Performance & Error Rates (Held-Out Evaluation)", self.heading_style))
        bench_data = [
            [
                Paragraph("<b>Evaluation Metric / Regime</b>", self.body_style),
                Paragraph("<b>Measured Value</b>", self.body_style),
                Paragraph("<b>Sample Size N</b>", self.body_style),
                Paragraph("<b>Reliability & Caveat</b>", self.body_style),
            ],
            [
                Paragraph("Fused PRISM AUC-ROC", self.body_style),
                Paragraph("<b>0.286</b>", self.mono_style),
                Paragraph("N = 9 (2 fake, 7 real)", self.body_style),
                Paragraph("[ANECDOTAL: N < 100, N_pos < 10]", self.body_style),
            ],
            [
                Paragraph("Fused PRISM EER", self.body_style),
                Paragraph("<b>0.714</b>", self.mono_style),
                Paragraph("N = 9 (2 fake, 7 real)", self.body_style),
                Paragraph("[ANECDOTAL: N < 100, N_pos < 10]", self.body_style),
            ],
            [
                Paragraph("Balanced Accuracy", self.body_style),
                Paragraph("<b>28.6%</b>", self.mono_style),
                Paragraph("N = 9 (2 fake, 7 real)", self.body_style),
                Paragraph("[ANECDOTAL: N < 100]", self.body_style),
            ],
            [
                Paragraph("High Conflict Subset ($K \\ge 0.5$)", self.body_style),
                Paragraph("<b>100.0% Error</b>", self.mono_style),
                Paragraph("N = 2", self.body_style),
                Paragraph("[ANECDOTAL: N < 100; triggers triage]", self.body_style),
            ],
            [
                Paragraph("Low Conflict Subset ($K < 0.2$)", self.body_style),
                Paragraph("<b>33.3% Error</b>", self.mono_style),
                Paragraph("N = 6", self.body_style),
                Paragraph("[ANECDOTAL: N < 100]", self.body_style),
            ],
        ]
        bench_table = Table(bench_data, colWidths=[140, 85, 125, 190])
        bench_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ]))
        story.append(bench_table)
        story.append(Spacer(1, 10))

        # Section 2: Visual Evidence Multi-Panel (if figure provided)
        if figure_png_bytes is not None:
            story.append(KeepTogether([
                Paragraph("2. Multi-Modal Spatial & Frequency Evidence Suite", self.heading_style),
                RLImage(io.BytesIO(figure_png_bytes), width=540, height=330),
                Spacer(1, 10),
            ]))

        # Section 3: Certified Forensic Reasoning Report
        story.append(Paragraph("3. Forensic Intelligence Laboratory Technical Reasoning", self.heading_style))
        clean_report = llm_report_text.replace("\n", "<br/>")
        story.append(Paragraph(clean_report, self.body_style))
        story.append(Spacer(1, 12))

        # Sign-off Footer
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#94a3b8"), spaceAfter=6))
        story.append(Paragraph(
            "<b>Investigative Support Notice:</b> This document was algorithmically compiled by the PRISM Automated Forensics Engine. "
            "This report is intended exclusively for investigative hypothesis generation and triage. It does not constitute definitive judicial testimony "
            "or court-admissible proof without independent verification by a certified forensic examiner.",
            self.subtitle_style,
        ))

        doc.build(story)
        pdf_buffer.seek(0)
        return pdf_buffer.getvalue()


# ---------------------------------------------------------------------------
# Unit / Smoke Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    exporter = ForensicReportExporter()
    print("Testing ForensicReportExporter...")
    print("-" * 60)

    dummy_bytes = b"PRISM_TEST_IMAGE_BYTES_12345"
    dummy_masses = {
        "ViT_spatial": {"T": 0.92, "A": 0.02, "U": 0.06},
        "PRNU_sensor": {"T": 0.85, "A": 0.05, "U": 0.10},
        "Compression_ELA": {"T": 0.78, "A": 0.08, "U": 0.14},
        "Metadata_Provenance": {"T": 0.95, "A": 0.00, "U": 0.05},
        "EyeGaze_Biometric": {"T": 0.88, "A": 0.04, "U": 0.08},
    }
    dummy_report = "EVIDENCE LAB AUDIT:\n1. Visual Forgery: ViT localized facial boundary splicing with 92% confidence.\n2. Sensor Noise: PRNU pattern deviation detected.\n3. Conclusion: The file exhibits incontrovertible indicators of deepfake synthesis."

    pdf_out = exporter.generate_pdf(
        filename="suspect_evidence_001.png",
        file_bytes=dummy_bytes,
        verdict="TAMPERED",
        score=0.965,
        belief=0.942,
        plausibility=0.988,
        agent_masses=dummy_masses,
        llm_report_text=dummy_report,
    )

    print(f"Generated PDF Size: {len(pdf_out)} bytes")
    assert len(pdf_out) > 1000
    assert pdf_out.startswith(b"%PDF")
    print("-" * 60)
    print("ForensicReportExporter unit test PASSED successfully!")
