from pathlib import Path
import pandas as pd


class FileLoader:
    """Loads CSV or Excel files into a pandas DataFrame."""

    @staticmethod
    def load(file) -> pd.DataFrame:
        """
        Load a CSV or Excel file.

        Parameters
        ----------
        file : str | Path | UploadedFile
            Local file path or Streamlit UploadedFile.

        Returns
        -------
        pd.DataFrame
        """

        # -----------------------------
        # Streamlit Uploaded File
        # -----------------------------
        if hasattr(file, "read"):

            # Always reset file pointer
            file.seek(0)

            filename = file.name.lower()

            try:
                if filename.endswith(".csv"):
                    return pd.read_csv(file)

                elif filename.endswith((".xlsx", ".xls")):
                    return pd.read_excel(file)

                else:
                    raise ValueError(
                        "Unsupported file format. Please upload a CSV or Excel file."
                    )

            except Exception as e:
                raise ValueError(f"Error reading uploaded file: {e}")

        # -----------------------------
        # Local File Path
        # -----------------------------
        path = Path(file)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        try:
            if path.suffix.lower() == ".csv":
                return pd.read_csv(path)

            elif path.suffix.lower() in [".xlsx", ".xls"]:
                return pd.read_excel(path)

            else:
                raise ValueError(
                    "Unsupported file format. Only CSV and Excel files are supported."
                )

        except Exception as e:
            raise ValueError(f"Error reading file: {e}")