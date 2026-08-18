"""
Dataset generator and preprocessor for Medical Specialty Classification.
Synthesizes clinical reports across 6 core medical specialties with stratified train/validation/test splits.
"""

import os
import random
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split

SPECIALTIES = [
    "Cardiovascular / Pulmonary",
    "Neurology",
    "Gastroenterology",
    "Orthopedics",
    "Oncology",
    "Endocrinology"
]

SPECIALTY_TEMPLATES = {
    "Cardiovascular / Pulmonary": [
        "Patient presents with sub-sternal chest discomfort radiating to left arm. ECG demonstrates sinus tachycardia at 104 bpm with non-specific T-wave inversions in leads V3-V5. Troponin I elevated at 0.45 ng/mL. Echocardiography shows preserved LVEF of 58%.",
        "Clinical history of progressive dyspnea on exertion and mild lower extremity edema. Auscultation reveals bilateral basilar crackles. Chest X-ray indicates cardiomegaly and mild pulmonary vascular congestion. Assessment: Acute decompensated heart failure.",
        "Follow-up for essential hypertension and coronary artery disease. Resting BP 142/88 mmHg, HR 76 bpm. Electrocardiogram reveals normal sinus rhythm without acute ischemic changes. Continue Lisinopril and Atorvastatin.",
        "Patient complains of palpitations and mild lightheadedness. Holter monitor summary reveals episodes of paroxysmal atrial fibrillation. Recommending anticoagulation evaluation and rate control therapy."
    ],
    "Neurology": [
        "Patient evaluated for persistent left-sided hemicranial headache accompanied by photophobia and nausea. Neurological examination reveals intact cranial nerves II-XII, normal tone, and 5/5 motor strength throughout. MRI Brain reveals minor non-specific T2/FLAIR white matter hyperintensities.",
        "Acute onset right-sided facial droop and mild expressive dysphasia noted 2 hours ago. Non-contrast CT head demonstrates no acute intracranial hemorrhage. Brain MRI shows acute ischemic infarct in left middle cerebral artery territory.",
        "Electroencephalogram (EEG) requested for spell evaluation. Background activity shows organized 9 Hz alpha rhythm with occasional temporal theta sharp waves. Assessment: Focal cortical excitability.",
        "Patient presents with resting tremor in right hand, bradykinesia, and rigidity. Gait exhibits reduced arm swing. Clinical assessment consistent with early Parkinsonian syndrome."
    ],
    "Gastroenterology": [
        "Patient reports epigastric burning pain aggravated by food intake. Esophagogastroduodenoscopy (EGD) reveals patchy mucosal erythema in gastric antrum. Gastric biopsy positive for Helicobacter pylori gastritis.",
        "Routine colonoscopy screening performed. Bowel preparation excellent. Inspection revealed two 4mm pedunculated polyps in sigmoid colon, completely resected via cold snare. Pathology pending.",
        "Right upper quadrant abdominal pain postprandially. Abdominal ultrasound shows diffuse increase in hepatic echogenicity consistent with Grade 2 hepatic steatosis (fatty liver disease). Liver enzymes: ALT 68 U/L, AST 54 U/L.",
        "Evaluation for chronic diarrhea and weight loss. Serum anti-tissue transglutaminase IgA elevated at 84 U/mL. Small bowel biopsy confirms villous atrophy consistent with Celiac disease."
    ],
    "Orthopedics": [
        "Patient fell on extended left arm. X-ray of left wrist demonstrates non-displaced fracture of distal radius. Closed reduction not required. Immobilized in short arm cast for 6 weeks.",
        "Evaluation of chronic right knee pain. Physical examination shows joint line tenderness and positive McMurray test. Knee MRI demonstrates complex tear of posterior horn of medial meniscus. Recommend physical therapy.",
        "Severe lower back pain radiating to L5 dermatome. Straight leg raise positive on right at 45 degrees. Lumbar spine MRI confirms L4-L5 right paracentral disc herniation compressing descending L5 nerve root.",
        "Post-operative follow-up for total hip arthroplasty. Surgical incision well healed without erythema or discharge. Plain radiographs demonstrate stable component positioning without osteolysis."
    ],
    "Oncology": [
        "Follow-up CT chest, abdomen, and pelvis for staging of pulmonary lesion. Scan shows stable 1.8cm solitary pulmonary nodule in right upper lobe without mediastinal lymphadenopathy. PET scan demonstrates low metabolic activity.",
        "Breast biopsy result review. Core needle biopsy of upper outer quadrant lesion demonstrates invasive ductal carcinoma, ER positive, PR positive, HER2 negative. Referred to surgical oncology multidisciplinary clinic.",
        "Routine post-treatment surveillance for Stage II colon adenocarcinoma. Carcinoembryonic antigen (CEA) level normal at 1.2 ng/mL. Surveillance CT scan demonstrates no recurrent disease or distant metastases.",
        "Bone marrow biopsy evaluation for unexplained cytopenias. Aspirate reveals hypercellular marrow with erythroid hyperplasia and dysplasia in 15% of cells. Cytogenetics pending."
    ],
    "Endocrinology": [
        "Patient with type 2 diabetes mellitus presents for routine check-up. Hemoglobin A1c is 8.2% (elevated). Fasting plasma glucose 164 mg/dL. Urine albumin-to-creatinine ratio within normal limits. Titrating Metformin.",
        "Evaluation for anterior neck fullness and fatigue. Thyroid ultrasound demonstrates a 1.2cm well-circumscribed hypoechoic nodule in left lobe. TSH elevated at 6.8 mIU/L, free T4 low at 0.7 ng/dL. Primary hypothyroidism.",
        "Assessment of adrenal incidentaloma noted on abdominal CT. 24-hour urinary free cortisol and plasma dexamethasone suppression test are normal. Non-functioning adrenal adenoma; plan repeat imaging in 12 months.",
        "Patient reports polyuria, polydipsia, and unintentional 5 kg weight loss. Random plasma glucose 285 mg/dL. Serum ketones negative. Initiating basal insulin regimen and diabetes self-management education."
    ]
}

def generate_dataset(samples_per_class: int = 45) -> Tuple[List[str], List[str]]:
    """Generates synthetic medical reports with corresponding specialty labels."""
    texts = []
    labels = []
    random.seed(42)

    for specialty, templates in SPECIALTY_TEMPLATES.items():
        for i in range(samples_per_class):
            base = random.choice(templates)
            # Add minor clinical variation
            variation = f" [Record Ref #{1000 + i}] Patient age: {random.randint(25, 82)}. Vital status stable."
            texts.append(base + variation)
            labels.append(specialty)

    return texts, labels

def get_stratified_splits(test_size: float = 0.2, val_size: float = 0.1) -> Dict[str, Tuple[List[str], List[str]]]:
    """Returns stratified train, validation, and test datasets."""
    texts, labels = generate_dataset()
    
    # First split into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    # Calculate adjusted val size relative to train+val
    adjusted_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=adjusted_val_size, random_state=42, stratify=y_train_val
    )
    
    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test)
    }
