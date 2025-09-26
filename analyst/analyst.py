"""Data analyst agent for automated analysis and reporting."""

# Standard library imports
import os
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from io import StringIO
from typing import Annotated

# Third-party imports
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
load_dotenv()

# Initializing the model
model = OpenAIModel('gpt-4.1', provider=OpenAIProvider(api_key=os.getenv('OPENAI_API_KEY')))

# Defining the state
@dataclass
class State:
    user_query: str = field(default_factory=str)
    file_name: str = field(default_factory=str)


# Defining the tools

def get_column_list(
    file_name: Annotated[str, "The name of the csv file that has the data"]
):
    """
    Use this tool to get the column list from the CSV file.

    Parameters:
    - file_name: The name of the CSV file that has the data
    """
    df = pd.read_csv(file_name)
    columns = df.columns.tolist()
    return str(columns)

# Getting the description of the column
def get_column_description(
    column_dict: Annotated[
        dict, "The dictionary of the column name and the description of the column"
    ]
):
    """
    Use this tool to get the description of the column.
    """

    return str(column_dict)

# Generating the graph
def graph_generator(
    code: Annotated[str, "The python code to execute to generate visualizations"]
) -> str:
    """
    Use this tool to generate graphs and visualizations using python code.

    Print the graph path in html and png format in the following format:
    'The graph path in html format is <graph_path_html> and the graph path in png
    format is <graph_path_png>'.

    """


    catcher = StringIO()

    try:
        with redirect_stdout(catcher):
            # The compile step can catch syntax errors early
            compiled_code = compile(code, '<string>', 'exec')
            exec(compiled_code, globals(), globals())

            return (
                f"The graph path is \n\n{catcher.getvalue()}\n"
                f"Proceed to the next step"
            )

    except Exception as e:
        return f"Failed to run code. Error: {repr(e)}, try a different approach"



# Executing the python code
def python_execution_tool(
    code: Annotated[str, "The python code to execute for calculations and data processing"]
) -> str:
    """
    Use this tool to run python code for calculations, data processing, and metric computation.

    Always use print statement to print the result in format:
    'The calculated value for <variable_name> is <calculated_value>'.

    """

    catcher = StringIO()

    try:
        with redirect_stdout(catcher):
            # The compile step can catch syntax errors early
            compiled_code = compile(code, '<string>', 'exec')
            exec(compiled_code, globals(), globals())

            return (
                f"The calculated value is \n\n{catcher.getvalue()}\n"
                f"Make sure to include this value in the report\n"
            )

    except Exception as e:
        return f"Failed to run code. Error: {repr(e)}, try a different approach"


# Defining the analyst agent

class AnalystAgentOutput(BaseModel):
    analysis_report: str = Field(description="The analysis report in markdown format")
    metrics: list[str] = Field(description="The metrics of the analysis")
    image_html_path: str = Field(
        description="The path of the graph in html format, if no graph is generated, return empty string"
    )
    image_png_path: str = Field(
        description="The path of the graph in png format, if no graph is generated, return empty string"
    )
    conclusion: str = Field(description="The conclusion of the analysis")

analyst_agent = Agent(
    model=model,
    tools=[
        Tool(get_column_list, takes_ctx=False),
        Tool(get_column_description, takes_ctx=False),
        Tool(graph_generator, takes_ctx=False),
        Tool(python_execution_tool, takes_ctx=False),
    ],
    deps_type=State,
    output_type=AnalystAgentOutput,
    instrument=True
)

# Defining the system prompt
@analyst_agent.system_prompt
async def get_analyst_agent_system_prompt(ctx: RunContext[State]):

    prompt = f"""
    You are an expert data analyst agent responsible for executing comprehensive data analysis workflows and generating professional analytical reports.

    **Your Mission:**
    Analyze the provided dataset to answer the user's query through systematic data exploration, statistical analysis, and visualization. Deliver actionable insights through a comprehensive report backed by quantitative evidence.

    **Available Tools:**
    - `get_column_list`: Retrieve all column names from the dataset
    - `get_column_description`: Get detailed descriptions and metadata for specific columns
    - `graph_generator`: Create visualizations (charts, plots, graphs) and save them in HTML and PNG formats. Use plotly express library to make the graph interactive.
    - `python_execution_tool`: Execute Python code for statistical calculations, data processing, and metric computation

    **Input Context:**
    - User Query: {ctx.deps.user_query}
    - Dataset File Name: {ctx.deps.file_name}

    **Execution Workflow:**
    **CRITICAL**: State is not persistent between tool calls. Always reload the dataset and import necessary libraries in each Python execution.

    1. **Dataset Discovery**: Use `get_column_list` to retrieve all available columns, then use `get_column_description` to understand column meanings and data types.

    2. **Analysis Planning**: Based on the user query and dataset structure, create a systematic analysis plan identifying:
       - Key variables to examine
       - Statistical methods to apply
       - Visualizations to create
       - Metrics to calculate

    3. **Data Exploration**: Load the dataset using pandas and perform initial exploration:
       - Check data shape, types, and quality
       - Identify missing values and outliers
       - Generate descriptive statistics

    4. **Statistical Analysis**: Execute the planned analysis using appropriate statistical methods:
       - Calculate relevant metrics and aggregations
       - Perform hypothesis testing if applicable
       - Identify patterns, trends, and correlations

    5. **Visualization Creation**: Generate meaningful visualizations that support your findings:
       - Use appropriate chart types for the data
       - Ensure visualizations are clear and informative
       - Save outputs in both HTML and PNG formats

    6. **Report Synthesis**: Compile all findings into a comprehensive analytical report.

    **Tool Usage Best Practices:**

    **python_execution_tool**:
    - Always include necessary imports: `import pandas as pd`, `import numpy as np`, `import matplotlib.pyplot as plt`, `import seaborn as sns`
    - Load dataset fresh each time: `df = pd.read_csv('{ctx.deps.file_name}')`
    - Use descriptive variable names and clear print statements
    - Format output: `print(f"The calculated value for {{metric_name}} is {{value}}")`
    - Handle errors gracefully with try-except blocks

    **graph_generator**:
    - Always include necessary imports and dataset loading
    - Create publication-quality visualizations with proper labels, titles, and legends
    - Save graphs using: `plt.savefig('graph.png', dpi=300, bbox_inches='tight')` and HTML equivalent
    - Print file paths in the required format: `print("The graph path in html format is <path.html> and the graph path in png format is <path.png>")`

    **get_column_list & get_column_description**:
    - Use these tools first to understand the dataset structure
    - Reference column information when planning analysis steps

    **Output Requirements:**
    Your final output must include:

    - **analysis_report**: Professional markdown report containing:
      * Executive Summary (key findings in 2-3 sentences)
      * Dataset Overview (structure, size, key variables)
      * Methodology (analytical approach taken)
      * Detailed Findings (organized by analysis steps)
      * Statistical Results (with proper interpretation)
      * Data Quality Assessment (missing values, outliers, limitations)
      * Insights and Implications

    - **metrics**: List of all calculated numerical values with descriptive labels

    - **image_html_path**: Path to HTML visualization file (empty string if none generated)

    - **image_png_path**: Path to PNG visualization file (empty string if none generated)

    - **conclusion**: Concise summary with actionable recommendations

    **Quality Standards:**
    - Use professional, data-driven language
    - Provide statistical context and significance levels
    - Explain methodologies and any assumptions made
    - Include confidence intervals where appropriate
    - Reference specific data points and calculated metrics
    - Format with proper markdown structure (headers, lists, tables, code blocks)
    - Ensure reproducibility by documenting all steps

    **Error Handling:**
    - If code execution fails, analyze the error and try alternative approaches
    - Handle missing data appropriately (document and address)
    - Validate results for reasonableness before reporting

    **Final Note:**
    Approach this analysis systematically. Think step-by-step, validate your work, and ensure every insight is backed by quantitative evidence. Your goal is to provide the user with a thorough, professional analysis that directly addresses their query.
    """
    return prompt



# Running the full agent
def run_full_agent(user_query: str, dataset_path: str):

    state = State(user_query=user_query, file_name=dataset_path)
    response = analyst_agent.run_sync(deps=state)
    print(response)
    response_data = response.data
    return response_data
