from langchain_core.prompts import ChatPromptTemplate

from src.llm.factory import LLMFactory

def test_llm():
    context = """
    -- CHUNK 1, from document: TrackWizz API Document Quality Check API --
    ENDPOINTS: /api/match/doc/v3
    METHODS: POST

    RAW CONTENT:
    Note: At least one of the POA is required. If details for permanent or correspondence address OVD type is provided the OVD number and OVD image for the same is to be provided.

    - Request and Response Structure:
    ==================================

    Content-Type: application/json


    -- CHUNK 2, from document: TrackWizz API Document Quality Check API --
    ENDPOINTS: /api/match/doc/v3
    METHODS: POST

    RAW CONTENT:
    1.Overview

    The API A108X can be used to verify the documents provided to the FI by the customers. FI can verify the OVDs according to the Feature Code provided in the API. The purpose for the Feature Code is to receive API responses according to the specific needs and verification requirements of FI. It ensures structured and flexible integration.

    The description of each Feature Code is as follows:


    -- CHUNK 3, from document: TrackWizz API Document Quality Check API --
    ENDPOINTS: /api/match/doc/v3
    METHODS: POST

    RAW CONTENT:
    Note: The Feature of masking with Feature Code 3 can be used when the masking is required for images given in an array of "aadhaarImages". For remaining OVD images, given in perPOA or corPOA masking will be done as usual in the "croppedOVDImage" field.


    -- CHUNK 4, from document: TrackWizz API Document Quality Check API --
    ENDPOINTS: /api/match/doc/v3
    METHODS: POST
    PURPOSE: REQUEST_BODY
    MANDATORY FIELDS: name, featureCode

    RAW CONTENT:
    2.2 Request Data Fields
    Field Name / JSON Tag | Mandatory | Data Type | Remarks
    name | Yes | String | Full Name of the Customer. (First Name / Middle Name / Last Name)  Yes if there is a need for comparison of data to be provided against OVD.
    fatherName | No | String | Full Name of the Customer’s  Father
    spouseName | No | String | Full Name of the Customer’s  Spouse
    gender | No | String | Gender of the Customer  (Male / Female / Transgender/Other)
    dob | No | String | Date of Birth of the Customer in DD-MM-YYYY format.
    emaild | No | String | Email ID of the Customer
    mobileNo | No | String | Mobile Number of the Customer
    corPOAImage | No | String  Enumerations | Base64 image string of the Correspondence Proof of Address. Images format provided in Enumerations.
    corPOAOvdType | No | String  Enumerations | Correspondence  address OVD type from  provided enumerations
    corPOAOvdNumber | No | String | Correspondence address OVD ID number
    corAddress | No | String | Correspondence Address details
    corCity | No | String | Correspondence Address City
    corState | No | String | Correspondence Address State
    corCountry | No | String | Correspondence Address Country
    corPin | No | String | Correspondence Address PinCode
    perPOAImage | No | String  Enumerations | Base64 image string of the Permanent Proof of Address. Images format provided in Enumerations.
    perPOAOvdType | No | String  Enumerations | Permanent address OVD type from provided in enumerations.
    perPOAOvdNumber | No | String | Permanent address OVD ID number
    perAddress | No | String | Permanent Address details
    perCity | No | String | Permanent Address City
    perState | No | String | Permanent Address State
    perCountry | No | String | Permanent Address Country
    perPin | No | String | Correspondence Address Pin Code
    photoImage | No | String  Enumerations | Base64 image string of the Customer Photograph. Supported Images format provided in Enumerations.
    aadhaarImages | No | Array | Aadhaar images or documents related to the the customer
    image | No | String  Enumerations | Base64 image string.Supported Images format provided in Enumerations.
    imageId | No | String  Enumerations | Base64 image string.Supported Images format provided in Enumerations.
    panCardImage | No | String  Enumerations | Base64 image string of Pancard of the customer
    panCardNumber | No | String | Pancard number of the customer.
    corPOAImageMaxSizeInK | No | Float | Maximum size allowed in corPOA image
    perPOAImageMaxSizeInKB | No | Float | Maximum size allowed in perPOA image
    panCardImageMaxSizeInKB | No | Float | Maximum size allowed in Pancard image
    featureCode | Yes | String | Feature code according to output required to be separated by ",". (Eg, "1,2")


    -- CHUNK 5, from document: TrackWizz API Document Quality Check API --
    ENDPOINTS: /api/match/doc/v3
    METHODS: POST

    RAW CONTENT:
    4.2.2 Correspondence POA Object Fields
    Field Name | Data Type | Remarks
    corPOAOvdType | String  (Enumerations) | Permanent Proof of Address OVD Type mentioned in Request.  Possible Values mentioned in Table
    classifiedOvdType | String  Enumerations | Classified Permanent Proof of Address OVD Type.  Possible values mentioned in Table.
    ovdNumberExtracted | String | OVD Number extracted
    ovdNumberConfidence | String  Enumerations | Possible values mentioned in Table.
    ovdNumberExtractionConfidenceScore | Float | Confidence score for OVD number extraction
    nameExtracted | String | Full name extracted from OVD
    nameConfidence | String  Enumerations | Possible values mentioned in Table.
    nameExtractionConfidenceScore | Float | Confidence score for name extracted
    dobExtracted | String | Date of Birth extracted from OVD
    dobConfidence | String  Enumerations | Possible values mentioned in Table.
    dobExtractionConfidenceScore | Float | Confidence score for DOB extracted
    genderExtracted | String | Gender extracted from OVD
    genderconfidence | String  Enumerations | Possible values mentioned in Table.
    genderExtractionConfidenceScore | Float | Confidence score for gender extracted
    spouseNameExtracted | String | Spouse Name extracted from OVD
    spouseNameConfidence | String  Enumerations | Possible values mentioned in Table.
    spouseNameExtractionConfidenceScore | Float | Confidence score for Spouse name extracted
    fatherNameExtracted | String | Father Name extracted from OVD
    fatherNameConfidence | String  Enumerations | Possible values mentioned in Table.
    fatherNameExtractionConfidenceScore | Float | Confidence score for Father name extracted
    addressExtracted | String | Address extracted from OVD
    addressConfidence | String  Enumerations | Possible values mentioned in Table.
    addressExtractionConfidenceScore | Float | Confidence score for address extracted
    countryExtracted | String | Passport issuing country or nationality extracted from Passport. Applicable only for passports.
    countryConfidence | String  Enumerations | Possible values mentioned in Table.
    passportCountryExtractionConfidenceScore | Float | Confidence score for country extracted
    pinExtracted | String | Postal code extracted from OVD
    pinConfidence | String  Enumerations | Possible values mentioned in Table.
    pinExtractionConfidenceScore | Float | Confidence score for pin extracted
    stateExtracted | String | State name extracted from OVD
    stateConfidence | String  Enumerations | Possible values mentioned in Table.
    cityExtracted | String | City name extracted from OVD
    cityConfidence | String  Enumerations | Possible values mentioned in Table.
    derivedStateExtracted | String | State derived by system using Pincode or address intelligence
    derivedStateConfidence | String  Enumerations | Possible values mentioned in Table.
    derivedCityExtracted | String | City derived by system using Pincode or address parsing logic
    derivedCityConfidence | String  Enumerations | Possible values mentioned in Table.
    emailExtracted | String | Email address extracted from OVD or supporting document if available
    emailConfidence | String  Enumerations | Possible values mentioned in Table.
    mobileNumberExtracted | String | Mobile number extracted from OVD or documents provided if available
    mobileNumberConfidence | String  Enumerations | Possible values mentioned in Table
    expiryDateExtracted | String | Expiry date extracted from OVD in YYYY-MM-DD format if available
    isExpired | Boolean | Provides output as true/false if expired.
    croppedOvdImage | String | Cropped image of the OVD provided
    croppedPhotoImage | String | Cropped face image extracted from OVD document (Masked if Aadhaar provided)
    photoMatchConfidence | String  Enumerations | Possible values mentioned in Table.
    compressedOvdImage | String | Compressed image of the OVD
    ovdImageSize | String | Size of the OVD image provided
    compressedOvdImageSize | String | Size of OVD image compressed
    compressedOvdImageStatusCode | String Enumerations | Possible values mentioned in Table


    -- CHUNK 6, from document: Untitled Document --
    PURPOSE: SUCCESS_RESPONSE
    MANDATORY FIELDS: requestId

    RAW CONTENT:
    6.2 Response Sample
    6.2.1 Successful response
    {
    "success": true,
    "message": "Quality check performed successfully",
    "data": {
    "perPOA": {
    "perPOAOvdType": "AadhaarRegular",
    "classifiedOvdType": "AadhaarFront",
    "ovdNumberExtracted": "955784251970",
    "ovdNumberConfidence": "Low",
    "ovdNumberExtractionConfidenceScore": 99.98,
    "nameExtracted": "Tapan Subhash Kocharekar",
    "nameConfidence": "Full_Mismatch",
    "nameExtractionConfidenceScore": 95.67,
    "dobExtracted": "21/Mar/1995",
    "dobConfidence": "Not_Applicable",
    "dobExtractionConfidenceScore": 99.44,
    "genderExtracted": "Male",
    "genderConfidence": "Not_Applicable",
    "genderExtractionConfidenceScore": 99.41,
    "addressConfidence": "NotRead",
    "pinConfidence": "NotRead",
    "croppedOvdImage": "base64",
    "croppedPhotoImage": "base64",
    "photoMatchConfidence": "Not_Applicable",
    "compressedOvdImage": ""base64"",
    "ovdImageSize": 46.26,
    "compressedOvdImageSize": 25.79,
    "compressedOvdImageStatusCode": "CM01"
    },
    "corPOA": {
    "corPOAOvdType": "AadhaarRegular",
    "classifiedOvdType": "AadhaarFront",
    "ovdNumberExtracted": "955784251970",
    "ovdNumberConfidence": "Low",
    "ovdNumberExtractionConfidenceScore": 99.98,
    "nameExtracted": "Tapan Subhash Kocharekar",
    "nameConfidence": "Full_Mismatch",
    "nameExtractionConfidenceScore": 95.67,
    "dobExtracted": "21/Mar/1995",
    "dobConfidence": "Not_Applicable",
    "dobExtractionConfidenceScore": 99.44,
    "genderExtracted": "Male",
    "genderConfidence": "Not_Applicable",
    "genderExtractionConfidenceScore": 99.41,
    "croppedOvdImage": "base64",
    "croppedPhotoImage": "base64",
    "photoMatchConfidence": "Not_Applicable",
    "compressedOvdImage": "base64",
    "ovdImageSize": 46.26,
    "compressedOvdImageSize": 25.79,
    "compressedOvdImageStatusCode": "CM01"
    },
    "pan": {
    "classifiedOvdType": "PanCard",
    "ovdNumberExtracted": "AAGPS8887D",
    "ovdNumberConfidence": "Low",
    "ovdNumberExtractionConfidenceScore": 99.44,
    "nameExtracted": "SAMIR SHASHIKANTSHAH",
    "nameConfidence": "Full_Mismatch",
    "nameExtractionConfidenceScore": 95.79,
    "dobExtracted": "6/Dec/1970",
    "dobConfidence": "Not_Applicable",
    "dobExtractionConfidenceScore": 98.61,
    "fatherNameExtracted": "SHASHIKANT SHAH",
    "fatherNameConfidence": "Extracted",
    "fatherNameExtractionConfidenceScore": 98.32,
    "croppedOvdImage": "base64",
    "croppedPhotoImage": "base64",
    "photoMatchConfidence": "Not_Applicable",
    "compressedOvdImage": "base64",
    "ovdImageSize": 47.64,
    "compressedOvdImageSize": 16.64,
    "compressedOvdImageStatusCode": "CM01"
    },
    "maskAadhaarImages": [
    {
    "maskImage": "base64",
    "imageId": "1",
    "aadhaarMaskedStatusCode": "AM01"
    }
    ]
    },
    "requestId": "UUID"
    }


    -- CHUNK 7, from document: Untitled Document --
    PURPOSE: SUCCESS_RESPONSE
    MANDATORY FIELDS: success, message, data, requestId

    RAW CONTENT:
    {
    "success": "boolean",
    "message": "string",
    "data": {
    "perPOA": {
    "perPOAOvdType": "string",
    "classifiedOvdType": "string",
    "ovdNumberExtracted": "string"
    "nameExtracted": "string",
    "dobExtracted": "string",
    "genderExtracted": "string",
    "addressExtracted": "string",
    "pinExtracted": "string",
    "stateExtracted": "string",
    "cityExtracted": "string",
    "derivedStateExtracted": "string",
    "derivedCityExtracted": "string",
    "emailExtracted": "string",
    "mobileNumberExtracted":"string",
    "passportCountryExtracted": "string",
    "expiryDateExtracted":"string",
    "croppedOVDImage" : "string"
    },
    "corPOA": {
    "corPOAOvdType": "string",
    "classifiedOvdType": "string",
    "ovdNumberExtracted": "string",
    "nameExtracted": "string",
    "dobExtracted": "string",
    "genderExtracted": "string",
    "addressExtracted": "string",
    "pinExtracted": "string",
    "stateExtracted": "string",
    "cityExtracted": "string",
    "derivedStateExtracted": "string",
    "derivedCityExtracted": "string",
    "emailExtracted": "string",
    "mobileNumberExtracted":"string",
    "passportCountryExtracted": "string",
    "expiryDateExtracted":"string",
    "croppedOVDImage" : "string"
    },
    "pan": {
    "classifiedOvdType": "string",
    "ovdNumberExtracted": "string",
    "nameExtracted": "string",
    "dobExtracted": "string",
    "fatherNameExtracted": "string",
    }
    },
    "requestId": "string"
    }

    ============================================================

    INFO:src.chat.engine:Generating response from LLM...
    "cityExtracted": "string",
    "derivedStateExtracted": "string",
    "derivedCityExtracted": "string",
    "emailExtracted": "string",
    "mobileNumberExtracted":"string",
    "passportCountryExtracted": "string",
    "expiryDateExtracted":"string",
    "croppedOVDImage" : "string"
    },
    "pan": {
    "classifiedOvdType": "string",
    "ovdNumberExtracted": "string",
    "nameExtracted": "string",
    "dobExtracted": "string",
    "fatherNameExtracted": "string",
    }
    },
    "requestId": "string"
    }

    ============================================================

    INFO:src.chat.engine:Generating response from LLM...
    "croppedOVDImage" : "string"
    },
    "pan": {
    "classifiedOvdType": "string",
    "ovdNumberExtracted": "string",
    "nameExtracted": "string",
    "dobExtracted": "string",
    "fatherNameExtracted": "string",
    }
    },
    "requestId": "string"
    }

    ============================================================

    INFO:src.chat.engine:Generating response from LLM...
    "fatherNameExtracted": "string",
    }
    },
    "requestId": "string"
    }
    """

    question = "According to the request rules in the A108X Quality Check API, what happens if an FI provides a corPOAOvdType value but leaves the OVD number or OVD image blank?"

    PROMPT = f"""
    You are an expert API Documentation Assistant in a Retrieval-Augmented Generation (RAG) system.

    You answer ONLY using the provided CONTEXT.

    ---

    # 1. STRICT GROUNDING RULES
    - Use ONLY information present in the CONTEXT.
    - Do NOT use external knowledge or invent scenarios.
    - Do NOT invent or fabricate missing endpoints, fields, parameters, schemas, or error codes.
    - LOGICAL DEDUCTION ALLOWANCE: You are explicitly allowed to perform direct, common-sense logical deductions and resolve synonyms based strictly on the context (e.g., understanding that a field being "blank", "omitted", or "empty" matches a documentation requirement stating it "must be provided" or is "missing").
    - If the answer is fundamentally missing or cannot be logically deduced from the text, respond exactly:
    "I don't have enough information in the provided documentation to answer that."

    ---

    # 2. CONTEXT INTERPRETATION RULES
    The CONTEXT may contain:
    - Headers (structure / grouping)
    - Code blocks / JSON (highest authority)
    - Tables (structured definitions)
    - Paragraphs (explanations)

    Priority order:
    1. Code / JSON
    2. Tables
    3. Headers
    4. Paragraphs

    Never merge unrelated endpoints or schemas.

    Treat structured data (tables, JSON, lists) as complete datasets. Note blocks or remarks immediately preceding or following a table apply directly to that table's rules.

    ---

    # 3. EXTRACTION RULE (HIGH PRIORITY)
    When the user asks for:
    - list
    - all
    - mandatory
    - required
    - fields
    - parameters

    Then:
    - scan ALL CONTEXT chunks completely
    - include EVERY valid match
    - do NOT stop early
    - do NOT summarize away missing rows

    ---

    # 4. EXPLANATION CONTROL RULE (CRITICAL)
    When the user asks "why", "how", or "what does it mean":

    - You MAY explain using ONLY information present in CONTEXT
    - You MAY reorganize and summarize for clarity
    - You MAY group related points for readability
    - BUT you MUST NOT:
    - add completely new external scenarios
    - introduce brand new error cases not found in the text
    - extend behavior beyond documentation

    Keep explanations concise and grounded.

    ---

    # 5. STRUCTURE PRESERVATION RULE
    - Do NOT distort or invent structure from CONTEXT
    - Do NOT rename fields or groups
    - Do NOT fabricate missing sections
    - Preserve original meaning as closely as possible

    ---

    # 6. MULTI-CHUNK RULES
    - If multiple endpoints exist, separate them clearly
    - Do not mix fields across endpoints
    - If conflicting information exists, explicitly mention ambiguity

    ---

    # 7. PARTIAL INFORMATION RULE
    - If only partial information exists:
    - clearly state what is present
    - clearly state what is missing
    - Never guess missing values

    ---

    # 8. RESPONSE STYLE RULE
    - Respond in clean Markdown
    - Be concise and technical by default
    - Use bullets for structured data
    - Use code blocks only when present in CONTEXT
    - Do NOT fabricate examples or enrich beyond CONTEXT

    ---

    # 9. SAFETY AGAINST HALLUCINATION
    - Never assume API behavior not explicitly stated or logically necessitated by the context
    - Never extend or complete missing documentation sections with outside knowledge

    """

    ASSISTANT_PROMPT = ChatPromptTemplate.from_messages([
        ("system", PROMPT),
        (
            "user",
            "CONTEXT:\n{context}\n\n"
            "USER QUESTION:\n{question}"
        )
    ])

    model = LLMFactory.get_model()

    chain = ASSISTANT_PROMPT | model

    result = chain.invoke({
        "context": context,
        "question": question
    })

    print(result.content)

if __name__ == "__main__":
    test_llm()