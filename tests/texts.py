"""Realistic multi-paragraph test texts for classifier/multi-chunk tests.

Written as actual prose (not repeated phrases) so chunk boundaries land mid-topic
and aggregation behaves as it would on real documents.
"""

INVOICE = """\
ACME Industrial Supply GmbH
Rechnung NR. 2026-0847
Steuer-Nr. 8/115/5789

Sehr geehrte Kundin, sehr geehrter Kunde,

vielen Dank für Ihre Bestellung vom 12. September 2026. Wie besprochen liefern
wir Ihnen die bestellten Komponenten für Ihre Fertigungsanlage in Kassel.

Position 1: 12x Industrie-Sensor Typ SH-45, Spannung 24V, je 189,00 EUR
Position 2: 3x Steuerkasten Modul EK-9, inkl. Montagesatz, je 1.240,00 EUR
Position 3: Installation und Inbetriebnahme vor Ort, Pauschale 980,00 EUR
Position 4: Schulung Ihrer Techniker (2 Tage), je Tag 690,00 EUR

Zwischensumme netto: 9.158,00 EUR
Umsatzsteuer 19%: 1.740,02 EUR
Gesamtbetrag: 10.898,02 EUR

Zahlbar ohne Abzug innerhalb von 14 Tagen nach Rechnungsdatum auf das unten
angegebene Konto. Bei Zahlung per SEPA-Lastschrift gewähren wir 2% Skonto.

Bankverbindung: Stadtbank Kassel, IBAN DE44 5123 0800 0004 5678 90,
BIC HELADEF1KAS.

Mit freundlichen Grüßen
Ihre ACME Industrial Supply GmbH
Maria Schneider, Abteilung Finanzen
"""

DONATION_RECEIPT = """\
Bescheinigung über Spende
Hilfswerk Grenzenlos e.V.
Spendenkonto bestätigt am: 03. Oktober 2026

Sehr geehrte Frau Weber,

herzlichen Dank für Ihre erneute Unterstützung unserer Bildungsprojekte in
Ostafrika. Ihre Spende in Höhe von 250,00 EUR ist am 30. September 2026 auf
dem Konto des Vereins eingegangen.

Der Hilfswerk Grenzenlos e.V. ist als gemeinnütziger Verein anerkannt
(Freistellungsbescheid des Finanzamts Bonn vom 15.03.2024, Aktenzeichen
63 08/5241). Die Spende wurde ausschließlich für steuerbegünstigte
wohltätige Zwecke im Sinne der Abgabenordnung verwendet.

Es wird bestätigt, dass es sich nicht um einen Mitgliedsbeitrag handelt und
 dass Sie für diese Zuwendung keine Gegenleistung erhalten haben.

Ihre Spende fließt in das Schulbau-Projekt in der Region Mwanza, das seit
2024 vier Grundschulen mit Unterrichtsmaterial, Sanitäranlagen und
Solartechnik ausgestattet hat. Der Bau der fünften Schule beginnt im
Januar 2027.

Bei Fragen zu dieser Bescheinigung wenden Sie sich gerne an unser
Spendenbüro. Diese Bestätigung gilt als vereinfachter Nachweis für das
Finanzamt bei Spenden bis 300 EUR.

Mit freundlichen Grüßen
Hilfswerk Grenzenlos e.V.
Thomas Krüger, Vorstand
"""

APARTMENT_LETTER = """\
Mietvertrag Wohngung 4.2, Bergmannstraße 18, 10961 Berlin

Zwischen der Hausverwaltung Neukölln GmbH (nachfolgend Vermieterin) und
Herrn Jonas Lindqvist (nachfolgend Mieter) wird folgender Mietvertrag
geschlossen:

1. Mietgegenstand: Die 3-Zimmer-Wohnung mit 78 qm im 4. Obergeschoss,
   inklusive Kellerabteil 6 und Stellplatz Nr. 14 im Innenhof.

2. Mietzweck: Wohnnutzung. Untervermietung bedarf der schriftlichen
   Zustimmung der Vermieterin.

3. Miete: Grundmiete 1.180,00 EUR monatlich, Betriebskostenvorauszahlung
   220,00 EUR, Gesamtzahlung 1.400,00 EUR fällig am dritten Werktag.

4. Dauer: Das Mietverhältnis beginnt am 01.11.2026 und wird auf
   unbestimmte Zeit geschlossen. Kündigungsfrist für den Mieter: drei
   Monate zum Monatsende.

5. Kaution: 2.800,00 EUR, zahlbar in drei Monatsraten, angelegt auf einem
   separaten Sparkonto der Vermieterin.

6. Schönheitsreparaturen: Der Mieter übernimmt die Durchführung der
   Schönheitsreparaturen während der Mietzeit nach den Fristen des
   Anhangs.

7. Haustiere: Kleintiere sind gestattet; Hunde nur mit Zustimmung.

Zusatzvereinbarung zum Balkon (Anbau 2027): Während der Bauarbeiten hat
der Mieter Anspruch auf eine Mietminderung von 10% für maximal vier
Monate.

Berlin, den 20. September 2026
Hausverwaltung Neukölln GmbH      Jonas Lindqvist
"""

# Mixed-domain documents for aggregation tests: early paragraphs point at one
# category, later paragraphs at another, like real papers whose abstract and
# conclusion differ in vocabulary.
PAPER_CS_HEAVY = """\
Learning Sparse Attention Patterns for Efficient Inference
Abstract: We propose a method for pruning attention matrices during inference.
Our algorithm learns which attention heads contribute to downstream accuracy
and drops the rest, reducing compute by a third with no measurable quality
loss on standard benchmarks. We evaluate on language modeling, code generation
and summarization tasks across three model scales.

The transformer architecture has become the default backbone for sequence
modeling, but its quadratic attention cost limits deployment on long inputs.
Prior work on sparse attention either fixes patterns heuristically or learns
them with reinforcement signals, both of which add training complexity.
We instead observe that head importance can be estimated from activation
statistics collected during a single calibration pass, requiring no gradient
information whatsoever.

Our pruning criterion ranks heads by the entropy of their attention
distributions: heads whose weights are near-uniform carry little routing
information and are safe to remove. We validate this hypothesis on three
families of models and find the correlation between head entropy and ablation
damage is stronger than for magnitude-based criteria.

Experiments: we fine-tune nothing. We load pretrained checkpoints, run the
calibration pass on 512 held-out sequences, prune, and measure perplexity.
Across all settings we match or exceed distillation baselines while being
roughly 40x cheaper to apply. Code and pruned checkpoints are released.
"""

PAPER_BIO_HEAVY = """\
Gut Microbiome Signatures Predict Immunotherapy Response in Melanoma
Abstract: The gut microbiome modulates systemic immunity, but its role in
checkpoint inhibitor response remains incompletely characterized. We profiled
stool metagenomes of 112 melanoma patients before anti-PD-1 therapy and
identified bacterial taxa whose abundance stratifies responders from
non-responders with an AUC of 0.83.

The intestinal microbiota shapes T-cell priming through short-chain fatty
acid production and dendritic cell modulation. Earlier smaller cohorts
reported associations between Akkermansia muciniphila and favorable outcomes,
but sample sizes limited statistical power and confounders such as antibiotic
exposure were inconsistently controlled.

We prospectively collected samples under a standardized protocol, recording
concomitant medication, diet proxies and proton-pump inhibitor use. After
adjustment, a five-taxon signature remained significant; Faecalibacterium
prausnitzii abundance was the strongest single predictor of progression-free
survival. Metagenomic functional analysis implicated tryptophan metabolism
pathways in the association.

These results suggest microbiome profiling could complement existing
biomarkers. Validation in an independent cohort of 89 patients reproduced
the signature's direction of effect, with attenuated effect size. Clinical
implementation requires interventional studies rather than observational
associations, and we outline a proposed trial design.
"""
