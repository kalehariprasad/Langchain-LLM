from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

# Define the splitter with desired configuration
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=700,
    chunk_overlap=0,
)

# Your Python code as a string
text = """
class DataHandler:
    def __init__(self, params_path: str):
        self.params_path = params_path
        self.params = self.load_params()

    def load_params(self) -> dict:
        \"\"\"Load parameters from a YAML file.\"\"\"
        try:
            with open(self.params_path, 'r') as file:
                params = yaml.safe_load(file)
            logging.info('Parameters retrieved from %s', self.params_path)
            return params
        except FileNotFoundError:
            logging.info('File not found: %s', self.params_path)
            raise
        except yaml.YAMLError as e:
            logging.info('YAML error: %s', e)
            raise
        except Exception as e:
            logging.info('Unexpected error: %s', e)
            raise CustomException(e, sys)

    def load_data(self, data_url: str) -> pd.DataFrame:
        \"\"\"Load data from a CSV file.\"\"\"
        try:
            df = pd.read_csv(data_url)
            logging.info('Data loaded from %s', data_url)
            # Calculate the percentage of null values in each column
            null_percentage = df.isnull().mean() * 100
            logging.info('Null value percentages:\\n%s', null_percentage)
            # Drop rows with null values if they are less than 5%
            if null_percentage.max() < 5:
                df = df.dropna()
                logging.info(
                    'Dropped rows with null values as they were less than 5% .'
                    )
            else:
                logging.warning(
                    'Null values exceed 5%, not dropping any rows.'
                    )
            return df
        except Exception as e:
            logging.info(
                'Unexpected error occurred while loading data: %s', e)
            raise CustomException(e, sys)

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Preprocess the data.\"\"\"
        try:
            pd.set_option('future.no_silent_downcasting', True)
            df = df.drop(columns=['tweet_id'], errors='ignore')
            final_df = df[df['sentiment'].isin(
                ['happiness', 'sadness']
            )].copy()
            final_df['sentiment'] = final_df['sentiment'].replace(
                {'happiness': 1, 'sadness': 0}
            )
            logging.info('Data preprocessing completed')
            return final_df
        except KeyError as e:
            logging.error('Missing column in the dataframe: %s', e)
            raise CustomException(e, sys)
        except Exception as e:
            logging.info('Unexpected error during preprocessing: %s', e)
            raise CustomException(e, sys)

    def save_object(
        self, object, file_path: str
    ) -> None:
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as file:
                pickle.dump(object, file)
        except Exception as e:
            logging.info(
                "unexpected error occured while saving object"
            )
            raise CustomException(e, sys)

    def save_data(
        self, data: pd.DataFrame, file_path: str
    ) -> None:
        \"\"\"Save the train and test datasets.\"\"\"
        try:
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            data.to_csv(file_path, index=False)
            logging.info('Processed data  saved to %s', file_path)
        except Exception as e:
            logging.info(
                'Unexpected error occurred while saving the data: %s', e)
            raise CustomException(e, sys)
"""

# Split the text
chunks = splitter.split_text(text)


print(len(chunks))
#print(chunks[4])


# Output the chunks
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i + 1} ---")
    print(chunk)
    print()
