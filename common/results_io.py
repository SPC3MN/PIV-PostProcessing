import os
import pandas as pd


def write_results_xlsx(rows, columns, path):
    """Write one row per case to `path` as a single Summary sheet, regenerating
    the whole workbook from `rows` each call.

    This replaces the old per-case pattern of reading an existing workbook,
    concatenating one row, and rewriting it inside the case loop (fragile: it
    assumed the file already existed with matching columns, and re-read/
    re-wrote the growing file once per case). Here the caller collects all
    rows in memory across the whole batch and calls this once at the end, so
    the workbook on disk always reflects exactly the cases just processed.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(rows, columns=columns)
    df.to_excel(path, index=False)
    return df
