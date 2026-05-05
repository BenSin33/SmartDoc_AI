CSS_STYLE = """
<style>
/* ===== 1. GLOBAL ===== */
    .stApp {
        background-color: #F8F9FA !important;
    }

    /* Fix toàn bộ text trong MAIN */
    [data-testid="stMain"],
    [data-testid="stMain"] * {
        color: #212529 !important;
    }

    /* ===== 2. SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background-color: #2C2F33 !important;
    }

    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }

    /* ===== 3. FILE UPLOADER ===== */
    [data-testid='stFileUploader'],
    [data-testid='stFileUploader'] * {
        color: #212529 !important;
    }

    [data-testid="stFileUploaderFileName"] {
        font-weight: bold !important;
    }

    /* ===== 4. BUTTON (FIX TOÀN BỘ) ===== */

    /* Button thường */
    button {
        background-color: #007BFF !important;
        color: #FFFFFF !important;
        border: none !important;
    }

    /* Button trong form (Hỏi) */
    div[data-testid="stForm"] button {
        background-color: #007BFF !important;
        color: #FFFFFF !important;
    }

    /* Hover */
    button:hover {
        background-color: #0056b3 !important;
        color: #FFFFFF !important;
    }

    /* ===== 5. INPUT ===== */
    input, textarea, [data-baseweb="input"] input {
        color: #FFFFFF !important;
        background-color: #3a3e44 !important;
        border-color: #555555 !important;
    }
    
    input::placeholder, textarea::placeholder {
        color: #999999 !important;
    }
    
    [data-baseweb="input"] {
        background-color: #3a3e44 !important;
    }

    /* ===== 5.1 SELECTBOX (FIX TEXT WHITE) ===== */
    /* ===== CONTAINER ===== */
    [data-baseweb="select"] {
        background-color: #3a3e44 !important;
        color: #FFFFFF !important;
    }

    /* ===== BUTTON ===== */
    [data-baseweb="select"] > div {
        background-color: #3a3e44 !important;
        border: 1px solid #555555 !important;
        border-radius: 8px !important;
    }

    /* ===== TEXT HIỂN THỊ (FIX LỖI CHÍNH) ===== */
    [data-baseweb="select"] div,
    [data-baseweb="select"] span {
        color: #FFFFFF !important;
    }

    /* ===== VALUE ĐANG CHỌN (QUAN TRỌNG NHẤT) ===== */
    [data-baseweb="select"] [class*="singleValue"] {
        color: #FFFFFF !important;
    }

    /* ===== INPUT ===== */
    [data-baseweb="select"] input {
        color: #FFFFFF !important;
        background-color: transparent !important;
    }

    /* ===== PLACEHOLDER ===== */
    [data-baseweb="select"] input::placeholder {
        color: #CCCCCC !important;
    }

    /* ===== DROPDOWN ===== */
    [data-baseweb="popover"] {
        background-color: #3a3e44 !important;
    }

    /* ===== MENU ===== */
    [data-baseweb="menu"] {
        background-color: #3a3e44 !important;
        border-radius: 8px !important;
        border: 1px solid #4a4e54 !important;
    }

    /* ===== ITEM ===== */
    [data-baseweb="menu"] li,
    [role="option"] {
        background-color: #3a3e44 !important;
        color: #FFFFFF !important;
    }

    /* ===== HOVER ===== */
    [data-baseweb="menu"] li:hover,
    [role="option"]:hover {
        background-color: #007BFF !important;
        color: #FFFFFF !important;
    }

    /* ===== SELECTED ===== */
    [aria-selected="true"] {
        background-color: #007BFF !important;
        color: #FFFFFF !important;
    }

    /* ===== FOCUS ===== */
    [data-baseweb="select"] > div:focus-within {
        border-color: #007BFF !important;
        box-shadow: 0 0 0 1px #007BFF !important;
    }

    /* ===== FIX TRIỆT ĐỂ (ANTI BUG STREAMLIT) ===== */
    [data-baseweb="select"] * {
        color: #FFFFFF !important;
    }

    /* ===== 6. MARK ===== */
    mark {
        background-color: #fff3cd;
        color: #212529 !important;
        padding: 2px 4px;
        border-radius: 4px;
    }
    /* Fix vùng drag & drop */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #FFFFFF !important;
        border: 2px dashed #CED4DA !important;
    }

    /* Text bên trong dropzone */
    [data-testid="stFileUploaderDropzone"] * {
        color: #212529 !important;
    }

    /* Hover cho đẹp */
    [data-testid="stFileUploaderDropzone"]:hover {
        background-color: #F1F3F5 !important;
    }
</style>
"""