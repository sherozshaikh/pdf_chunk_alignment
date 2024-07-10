from typing import List
import os
import re
import zipfile
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download(['punkt','stopwords'])
from nltk.tokenize import word_tokenize

class DocMapper():
  """
  A class to find close elements links between two documents.
  """
  def __init__(self,doc1_elements_list:List[str],doc2_elements_list:List[str],doc1_elements_embedding:np.ndarray=None,doc2_elements_embedding:np.ndarray=None,threshold_:float=0.6,output_folder:str='Mapped_Attributes',same_flag:bool=False):
    """
    Initialize the DocMapper class.

    Args:
    - doc1_elements_list (List[str]): List containing Doc1 Elements.
    - doc2_elements_list (List[str]): List containing Doc2 Elements.
    - doc1_elements_embedding (ndarray, optional): Doc1 Embeeding Vector.
    - doc2_elements_embedding (ndarray, optional): Doc2 Embeeding Vector.
    - threshold_ (float, optional), default = 0.6: Threshold value for filtering similarity scores.
    - output_folder (str), default = Mapped_Attributes: Output Folder Name.
    - same_flag (bool), default = False: If same then retrive only lower trianglular cosine matrix.

    """
    self.doc1_elements_list:List[str] = doc1_elements_list
    self.doc2_elements_list:List[str] = doc2_elements_list
    self.doc1_elements_embedding:np.ndarray = doc1_elements_embedding
    self.doc2_elements_embedding:np.ndarray = doc2_elements_embedding
    self.threshold_:float = threshold_
    self.output_folder:str = self.trim_characters(stxt=output_folder).replace(' ','_')
    self.same_flag:bool = same_flag

  def __repr__(self):
    """
    Returns a string representation of the class instance.
    """
    return f"DocMapper()"

  def __str__(self):
    """
    Returns a description of the class.
    """
    return "Class to fetch Similar Doc1 Elements for given Doc2 Elements."

  def trim_characters(self,stxt:str='')->str:
    """
    Removes non-alphanumeric characters from a string.

    Args:
    - stxt (str): Input string.

    Returns:
    - str: String with non-alphanumeric characters removed.
    """
    return re.compile(pattern=r'\s+').sub(repl=r' ',string=str(re.compile(pattern=r'[^a-zA-Z\d]').sub(repl=r' ',string=str(stxt)))).strip()

  def create_final_folder(self)->None:
    """
    Creates Output Folder.
    If the folder already exists, it is first removed along with all its contents, and then a new empty folder is created.

    Returns:
    - None
    """
    if os.path.exists(path=self.output_folder):
      os.rmdir(path=self.output_folder)
    os.mkdir(path=self.output_folder)
    return None

  def create_final_zip(self)->None:
    """
    Creates a ZIP archive of all the contents.
    This method walks through the directory structure, adds all files to a ZIP archive, and stores it as '.zip'.

    Returns:
    - None
    """
    # Creates ZIP
    with zipfile.ZipFile(file=self.output_folder+'.zip',mode='w',compression=zipfile.ZIP_DEFLATED) as zip_file:
      for all_root,all_dirs,all_files in os.walk(self.output_folder):
        for file_1 in all_files:
          temp_file_path = os.path.join(all_root,file_1)
          zip_file.write(
            temp_file_path,
            os.path.relpath(temp_file_path,self.output_folder)
            )

    zip_file_path:str = self.output_folder+'.zip'
    target_folder_path:str = self.output_folder
    os.rename(os.path.abspath(zip_file_path),os.path.abspath(os.path.join(target_folder_path,zip_file_path)))
    return None

  def pre_processing_text_values(self,txt:str='',is_lower:bool=True,remove_characters:bool=True)->str:
    """
    Pre-processes text values by lowercasing, removing non-alphanumeric characters, and tokenizing.

    Args:
    - txt (str): Input text.
    - is_lower (bool, optional), default = True: Convert text to lowercase.
    - remove_characters (bool, optional), default = True: Remove non-alphanumeric characters.

    Returns:
    - str: Pre-processed text.
    """
    if is_lower:
      txt:str=str(txt).lower().strip()
    else:
      txt:str=str(txt).strip()

    if remove_characters:
      txt:str=self.trim_characters(stxt=txt)
    else:
      pass

    return ' '.join([x for x in word_tokenize(txt) if x.isalnum()])

  def calculate_similarity_tfidf(self,texts1:List[str],texts2:List[str])->np.ndarray:
    """
    Calculates TF-IDF cosine similarity between two lists of texts.

    Args:
    - texts1 (List[str]): List of first texts.
    - texts2 (List[str]): List of second texts.

    Returns:
    - np.ndarray: Similarity score matrix.
    """
    tfidf_vectorizer = TfidfVectorizer(decode_error = 'strict',use_idf = True,smooth_idf = True,binary = False,lowercase = True,max_features = 30_000,dtype = np.float32,ngram_range = (1,4),stop_words = 'english').fit(texts1+texts2)
    texts1_matrix:np.ndarray = tfidf_vectorizer.transform(texts1)
    texts2_matrix:np.ndarray = tfidf_vectorizer.transform(texts2)
    similarity_score_matrix:np.ndarray = cosine_similarity(texts1_matrix,texts2_matrix)
    return similarity_score_matrix

  def calculate_similarity_score(self,texts1_matrix:np.ndarray,texts2_matrix:np.ndarray)->np.ndarray:
    """
    Calculates cosine similarity between two matrices of texts.

    Args:
    - texts1_matrix (np.ndarray): First matrix of texts.
    - texts2_matrix (np.ndarray): Second matrix of texts.

    Returns:
    - np.ndarray: Similarity score matrix.
    """
    return cosine_similarity(texts1_matrix,texts2_matrix)

  def filter_similarity_matrix(self,similarity_matrix:np.ndarray,threshold_val:float=0.65)->pd.DataFrame:
    """
    Filters similarity matrix based on a threshold value.

    Args:
    - similarity_matrix (np.ndarray): Similarity score matrix.
    - threshold_val (float, optional), default = 0.65: Threshold value for similarity. Update using "threshold_" during initialization

    Returns:
    - pd.DataFrame: Filtered DataFrame with relevant attributes and scores.
    """

    if self.same_flag:
      similarity_matrix:np.ndarray = np.tril(similarity_matrix,k=-1) # k=-1 excludes the diagonal
    else:
      pass

    relevant_indices:np.ndarray = np.argwhere(similarity_matrix > threshold_val)
    ids_1:np.ndarray = relevant_indices[:, 0]
    ids_2:np.ndarray = relevant_indices[:, 1]
    filtered_scores:np.ndarray = similarity_matrix[ids_1,ids_2]
    results:list = [{
        'doc1_elements': self.doc1_elements_list[i],
        'doc2_elements': self.doc2_elements_list[j],
        'Score': k,
    } for i, j, k in zip(ids_1, ids_2, filtered_scores)]

    results_df:pd.DataFrame = pd.DataFrame(data=results)
    results_df[['doc1_elements','doc2_elements']] = pd.DataFrame(np.sort(results_df[['doc1_elements','doc2_elements']],axis=1),index=results_df.index)
    results_df:pd.DataFrame = results_df.drop_duplicates(subset=['doc1_elements','doc2_elements'])
    results_df:pd.DataFrame = results_df.sort_values(by=['doc1_elements','Score'],ascending=[True,False])
    return results_df

  def main(self)->None:
    """
    Main function to perform attribute mapping and write results to a CSV file.
    """
    # Calculate similarity scores based on the availability of an embedding model
    if (self.doc1_elements_embedding is not None and np.any(self.doc1_elements_embedding != None)) and (self.doc2_elements_embedding is not None and np.any(self.doc2_elements_embedding != None)):
      similarity_score:np.ndarray = self.calculate_similarity_score(texts1_matrix=self.doc1_elements_embedding,texts2_matrix=self.doc2_elements_embedding)
    else:
      processed_doc1_elements_list:list=[self.pre_processing_text_values(txt=x,is_lower=True,remove_characters=True) for x in self.doc1_elements_list]
      processed_doc2_elements_list:list=[self.pre_processing_text_values(txt=x,is_lower=True,remove_characters=True) for x in self.doc2_elements_list]
      similarity_score:np.ndarray = self.calculate_similarity_tfidf(texts1=processed_doc1_elements_list,texts2=processed_doc2_elements_list)

    # Filter and process the similarity matrix
    mapped_result_df:pd.DataFrame = self.filter_similarity_matrix(similarity_matrix=similarity_score,threshold_val=self.threshold_)
    mapped_result_df['Score']:pd.Series = mapped_result_df['Score']*100
    mapped_result_df['Score']:pd.Series = mapped_result_df['Score'].round(4)

    # Finalize and write the results to a CSV file
    self.create_final_folder() # create folder
    mapped_result_df.to_csv(path_or_buf=self.output_folder+'/Mapping.csv',index=False,mode='w',encoding='utf-8') # save in CSV file format
    del mapped_result_df

    # self.create_final_zip() # create zip
    print('wrote the final output to local: ',time.ctime())
    return None

def custom_ram_cleanup_func()->None:
  """
  Clean up global variables except for specific exclusions and system modules.

  This function deletes all global variables except those specified in
  `exclude_vars` and variables starting with underscore ('_').

  Excluded variables:
  - Modules imported into the system (except 'sys' and 'os')
  - 'sys', 'os', and 'custom_ram_cleanup_func' itself

  Returns:
  None
  """

  import sys
  all_vars = list(globals().keys())
  exclude_vars = list(sys.modules.keys())
  exclude_vars.extend(['In','Out','_','__','___','__builtin__','__builtins__','__doc__','__loader__','__name__','__package__','__spec__','_dh','_i','_i1','_ih','_ii','_iii','_oh','exit','get_ipython','quit','sys','os','custom_ram_cleanup_func',])
  for var in all_vars:
      if var not in exclude_vars and not var.startswith('_'):
          del globals()[var]
  del sys
  return None

# Example usage:
if __name__ == "__main__":

  # Simulation Run

  # Fake Doc1
  doc1_elements:list=[
    "As part of our construction project, we will install solar panels on the rooftop to harness renewable energy and reduce our carbon footprint. This initiative aligns with our commitment to sustainability and will contribute significantly to our energy independence.",
    "Construction of a new office building with sustainable materials and energy-efficient systems to minimize environmental impact.",
    "Construction of a parking structure for employees.",
    "Construction of bicycle storage facilities to encourage eco-friendly transportation choices.",
    "Construction of new office building with sustainable materials.",
    "Construction of parking structure for employee convenience.",
    "Creation of outdoor recreational spaces to promote employee well-being and encourage physical activity.",
    "Design and construction of pedestrian-friendly pathways to enhance accessibility and safety.",
    "Design and implementation of a green roof to enhance biodiversity and mitigate urban heat island effects.",
    "Design and implementation of green roof for biodiversity.",
    "Design and installation of green walls to improve air quality and provide natural insulation.",
    "Implementation of energy-efficient HVAC systems to improve indoor air quality and reduce operational costs.",
    "Implementation of rainwater harvesting systems for water reuse and sustainability.",
    "Implementation of waste management solutions to reduce landfill waste and promote recycling efforts.",
    "Installation of electric vehicle charging stations to support sustainable commuting options.",
    "Installation of smart lighting solutions for energy savings and improved lighting quality.",
    "Installation of solar panels on the rooftop to harness renewable energy and reduce carbon footprint.",
    "Integration of digital security systems for enhanced building and data protection.",
    "Integration of fire safety systems to ensure building protection and occupant safety.",
    "Landscaping with native plants to promote environmental sustainability and enhance aesthetic appeal.",
    "Our scope of work involves renovating the existing electrical systems to meet current safety standards and enhance operational efficiency. We will implement advanced technologies to optimize energy consumption while ensuring reliability and compliance with regulatory requirements.",
    "Renovation of existing electrical systems to meet current safety standards and optimize energy consumption.",
    "The construction of a new parking structure for employees is essential to alleviate current parking shortages and improve accessibility. Our design focuses on maximizing space efficiency and incorporating eco-friendly materials to enhance the overall environmental impact of the project.",
    "Upgrade of building insulation to improve energy efficiency and maintain indoor comfort.",
    "Upgrade of elevators to meet accessibility standards and enhance user experience.",
    "Upgrade of plumbing infrastructure to enhance water conservation and ensure sustainable usage.",
    "We are tasked with designing and implementing a green roof atop the new office building to enhance biodiversity, improve air quality, and mitigate urban heat island effects. Our approach integrates innovative planting techniques and sustainable irrigation systems.",
    "We propose to construct a state-of-the-art office building equipped with sustainable materials and advanced energy-efficient systems to minimize environmental impact and enhance employee productivity. The project includes innovative design elements such as a green roof and solar panels integrated into the building's structure.",
  ]

  # Fake Doc2
  doc2_elements:list=[
    "Coverage for building insulation upgrades, protecting against energy loss and maintaining thermal efficiency.",
    "Coverage for elevator upgrades and installations, protecting against mechanical failures and ensuring safety compliance.",
    "Coverage for employee parking structure construction, protecting against accidents and liability claims during construction phases.",
    "Coverage for green wall design and installation, protecting against maintenance issues and structural damage.",
    "Coverage for liability during the construction of office buildings, protecting against property damage and third-party claims.",
    "Coverage for pedestrian-friendly pathway projects, protecting against accidents and ensuring safe access.",
    "Coverage for smart lighting solutions, protecting against electrical failures and ensuring continuous operation.",
    "Insurance for digital security system integration, covering data breaches and system failures impacting security measures.",
    "Insurance for electric vehicle charging stations, covering equipment damage and liability related to charging operations.",
    "Insurance for HVAC system installations, covering system failures and performance issues impacting operational efficiency.",
    "Insurance for landscaping projects with native plants, covering damage and loss due to natural disasters and accidents.",
    "Insurance for rainwater harvesting systems, covering equipment failures and water quality issues.",
    "Insurance for solar panel installation projects, covering equipment damage and performance issues due to adverse conditions.",
    "Insuring green roofs against damage from severe weather conditions, vandalism, and structural issues.",
    "Our insurance package includes coverage specifically tailored for the solar panel installation project, offering protection against equipment damage, performance issues, and financial losses due to adverse weather conditions or operational failures.",
    "Our insurance plan includes coverage for the green roof of the new office building, protecting against damage from severe weather conditions, vandalism, and structural issues. We emphasize proactive risk management to safeguard your sustainable investments.",
    "Our insurance policy provides comprehensive coverage for all liability risks during the construction phases of the new office building project. We ensure protection against property damage, third-party injuries, and unforeseen events to safeguard your investment.",
    "Policy for bicycle storage facilities construction, providing coverage for theft, damage, and liability claims.",
    "Policy for construction of outdoor recreational spaces, providing coverage for construction-related risks and liability.",
    "Policy for fire safety system integration, providing coverage for fire incidents and compliance-related liabilities.",
    "Policy for plumbing infrastructure upgrades, providing coverage for leaks, failures, and water damage.",
    "Policy for renovations and upgrades of electrical systems, providing coverage for equipment failures and business interruption.",
    "Policy for waste management solutions, providing coverage for environmental liabilities and waste disposal incidents.",
    "We offer specialized insurance solutions for the renovation of electrical systems, providing coverage for equipment failures, electrical fires, and business interruption. Our policies are designed to minimize financial risks and ensure business continuity.",
    "We provide comprehensive insurance coverage for the construction of employee parking structures, protecting against structural damage, accidents during construction, and liability claims. Our tailored solutions ensure peace of mind and financial security.",
  ]

  # Vector Embedding using Traditional Methods

  DocMapper(
    doc1_elements_list=doc1_elements,
    doc2_elements_list=doc2_elements,
    doc1_elements_embedding=None,
    doc2_elements_embedding=None,
    threshold_=0.7,
    output_folder='TF_IDF',
    same_flag=False,
    ).main()

  custom_ram_cleanup_func()
  del custom_ram_cleanup_func

