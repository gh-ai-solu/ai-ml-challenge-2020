import PyPDF3
import docx
import re


def read_EULAs(lst_doc_name):
    """
    Reads a list of documents and breaks apart each document

    Parameters
    ----------
    lst_doc_name: list of string
        list of input file paths

    Returns
    -------
    dict_output: dictionary
        a dictionary of individual clauses of the corresponding EULA
    """
    dict_output = {}
    for doc_name in lst_doc_name:
        if doc_name[-4:] == "docx":
            lst_output = parse_docx(doc_name)
        elif doc_name[-3:] == "pdf":
            lst_output = parse_pdf(doc_name)
        else:
            print("file you selected is not a supported \
                type (docx, pdf).\n" + doc_name)
            continue
        dict_output[doc_name] = lst_output
    return(dict_output)


def parse_docx(doc_name):
    """
    Reads a Microsoft Word document given filepath
    and divides it into individual clauses

    Parameters
    ----------
    doc_name: string
        filepath of the Microsoft Word document with docx extension

    Returns
    -------
    list of string
        an array of individual clauses of the EULA
    """
    document = docx.Document(doc_name)
    lst_output = []
    clause = ""
    for para in document.paragraphs:
        txt = para.text.strip()
        if para.text == "":
            continue
        clause = " ".join([clause, txt])
        clause = clause.strip()
        # if the paragraph does not end with a period,
        # combine it with the next one
        if txt[-1] == ".":
            lst_output.append(clause)
            clause = ""
    return(lst_output)


def parse_pdf(doc_name):
    """
    Reads an Adobe PDF document given filepath and
    divides it into individual clauses

    Parameters
    ----------
    doc_name: string
        filepath of the Adobe PDF document with pdf extension

    Returns
    -------
    list of string
        an array of individual clauses of the EULA
    """

    document = PyPDF3.PdfFileReader(doc_name)
    pageNum = document.getNumPages()
    lst_output = []
    clause = ""
    for int_page in range(pageNum):
        lst_text = document.getPage(int_page).extractText().split(" \n")

        for text in lst_text:
            # clean the text
            txt = text.strip()
            txt = re.sub('\n', '', txt)
            if txt == "":
                continue
            clause = " ".join([clause, txt])
            clause = clause.strip()
            if txt[-1] == "." and len(clause) > 30:
                lst_output.append(clause)
                clause = ""
    return(lst_output)
