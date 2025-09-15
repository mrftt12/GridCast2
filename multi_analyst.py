import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from typing import Annotated
import asyncio
from datetime import datetime
import re
from pydantic_graph import Graph, BaseNode, GraphRunContext, End
import os
import pandas as pd
from io import StringIO
from contextlib import redirect_stdout
load_dotenv()


model = OpenAIModel('gpt-4.1', provider=OpenAIProvider(api_key=os.getenv('OPENAI_API_KEY')))


@dataclass
class State:
    user_query: str = field(default_factory=str)
    file_name: str = field(default_factory=str)
    column_list: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    instructions: list[str] = field(default_factory=list)
    column_dict: dict[str, str] = field(default_factory=dict)
    analysis: list[str] = field(default_factory=list)
    conclusion: list[str] = field(default_factory=list)
    image_html_path: list[str] = field(default_factory=list)
    image_png_path: list[str] = field(default_factory=list)
    current_step_index: int = field(default=0)


# Tools

async def get_column_list(
    file_name: Annotated[str, "The name of the csv file that has the data"]
):
    """
    Use this tool to get the column list from the CSV file.
    
    Parameters:
    - file_name: The name of the CSV file that has the data
    """
    df = pd.read_csv(file_name)
    columns = df.columns.tolist()
    dtype = df.dtypes.to_dict()
    return str(f"The column list is \n{columns} and the data type is \n{dtype}")


# Generating the graph
def graph_generator(
    code: Annotated[str, "The python code to execute to generate visualizations"]
) -> str:
    """
    Use this tool to generate graphs and visualizations using python code.

    Print the graph path in html and png format in the following format:
    'The graph path in html format is <graph_path_html> and the graph path in png format is <graph_path_png>'.

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



# Building planner agent

class PlannerAgentOutput(BaseModel):
    steps: list[str] = Field(description="The steps to be followed to complete the analysis")
    instructions: list[str] = Field(description="The instructions for the next agent for each step")
    column_dict: dict[str, str] = Field(description="The dictionary of the column name and the description of the column")



planner_agent = Agent(
    model=model,
    deps_type=State,
    result_type=PlannerAgentOutput,
    tools=[Tool(get_column_list, takes_ctx=False)]
)

@planner_agent.system_prompt
async def get_planner_agent_system_prompt(ctx: RunContext[State]):
    prompt = f"""
    You are a strategic data analysis planner responsible for creating comprehensive analysis workflows.

    **Primary Objective:**
    Design a systematic, step-by-step analysis plan that transforms the user's query into actionable analytical tasks for execution agents.

    **Available Tools:**
    - get_column_list: Extract all column names and data types from the CSV dataset

    **Planning Methodology:**
    
    1. **Dataset Exploration**
       - Use get_column_list to understand the dataset structure
       - Identify key variables relevant to the user's query
       - Assess data types and potential analytical approaches

    2. **Query Analysis**
       - Decompose the user's request into specific analytical objectives
       - Identify required statistical methods and visualizations
       - Determine appropriate metrics and comparisons needed

    3. **Workflow Design**
       - Create 5-7 sequential analysis steps that build logically
       - Ensure each step contributes to answering the user's query
       - Plan for both exploratory and confirmatory analysis

    **Analysis Categories to Consider:**
    - **Descriptive Analytics**: Summary statistics, distributions, central tendencies
    - **Data Quality Assessment**: Missing values, outliers, data integrity checks
    - **Comparative Analysis**: Group comparisons, trend analysis, correlations
    - **Visual Analytics**: Charts, plots, and graphs that reveal patterns using plotly express
    - **Statistical Testing**: Hypothesis tests, significance testing when appropriate

    **Output Specifications:**
    - **steps**: Ordered list of clear, executable analysis tasks
    - **instructions**: Detailed implementation guidance for each step (methodology only, no code)
    - **column_dict**: Dictionary mapping relevant columns to their analytical purpose and description

    **Implementation Guidelines:**
    - Recommend specific pandas/numpy methods for data manipulation
    - Suggest appropriate visualization libraries (plotly express)
    - Include data preprocessing requirements (cleaning, filtering, transformations)
    - Specify statistical tests with justification for their selection
    - Consider data scale, distribution, and type when planning operations

    **Current Request Context:**
    User Query: {ctx.deps.user_query}
    Dataset: {ctx.deps.file_name}

    **Success Criteria:**
    Create a plan that is systematic, comprehensive, and directly addresses the user's analytical needs while leveraging the dataset's full potential for insights.
    provide the output in JSON format.
    """
    return prompt


# Execution Agent



class ExecutionAgentOutput(BaseModel):
    analysis_report: str = Field(description="The analysis report in markdown format")
    metrics: list[str] = Field(description="The metrics of the analysis")
    image_html_path: str = Field(description="The path of the graph in html format, if no graph is generated, return empty string")
    image_png_path: str = Field(description="The path of the graph in png format, if no graph is generated, return empty string")
    conclusion: str = Field(description="The conclusion of the analysis")

execution_agent = Agent(
    model=model,
    tools=[Tool(graph_generator), Tool(python_execution_tool)],
    deps_type=State,
    result_type=ExecutionAgentOutput
)

@execution_agent.system_prompt
async def get_execution_agent_system_prompt(ctx: RunContext[State]):
    
    prompt = f"""
    You are a data analysis execution specialist responsible for implementing analytical workflows and generating professional reports.

    **Core Mission:**
    Execute the specific analysis step provided by the planner, perform precise calculations, create meaningful visualizations, and deliver a focused analysis report for this step.

    **Available Tools:**
    - python_execution_tool: Execute Python code for data processing, statistical analysis, and calculations
    - graph_generator: Create interactive visualizations and save in HTML/PNG formats

    **Critical Execution Notes:**
    - **State Persistence**: No state carries between tool calls - always reload data and import libraries
    - **Data Loading**: Use `pd.read_csv('{ctx.deps.file_name}')` in every Python execution
    - **Library Imports**: Include all necessary imports (pandas, numpy, matplotlib, plotly, etc.) in each code block

    **Current Task Context:**
    - User Query: {ctx.deps.user_query}
    - Current Step: {ctx.deps.steps[ctx.deps.current_step_index] if ctx.deps.current_step_index < len(ctx.deps.steps) else "Final step"}
    - Step Instructions: {ctx.deps.instructions[ctx.deps.current_step_index] if ctx.deps.current_step_index < len(ctx.deps.instructions) else "Complete analysis"}
    - Available Columns: {ctx.deps.column_dict}
    - Dataset File: {ctx.deps.file_name}
    - Previous Analysis: {ctx.deps.analysis[-1] if ctx.deps.analysis else "No previous analysis"}

    **Execution Protocol:**
    
    1. **Data Preparation**
       - Load the dataset using pandas
       - Verify data integrity and structure
       - Apply necessary preprocessing based on step requirements

    2. **Analysis Implementation**
       - Execute the specific analytical task for this step
       - Use appropriate statistical methods and calculations
       - Handle missing data and outliers appropriately

    3. **Visualization Creation (if required)**
       - Generate charts that directly support the analysis
       - Use plotly express for interactive visualizations
       - Ensure proper labels, titles, and formatting
       - Save in both HTML and PNG formats

    4. **Results Documentation**
       - Calculate and document all relevant metrics
       - Provide statistical context and interpretation
       - Note any limitations or assumptions

    **Tool Usage Best Practices:**

    **python_execution_tool**:
    - Always start with: `import pandas as pd; import numpy as np; df = pd.read_csv('{ctx.deps.file_name}'); print(f"Previous analysis: {ctx.deps.analysis[-1] if ctx.deps.analysis else 'No previous analysis'}")`
    - Use clear variable names and add comments
    - Print results in format: `print(f"The calculated {{metric_name}} is {{value}}")`
    - Handle exceptions gracefully

    **graph_generator**:
    - Use plotly express for visualization to make it interactive
    - Include complete code with imports and data loading
    - Create publication-quality visualizations
    - Use consistent color schemes and formatting
    - Print paths: `print("The graph path in html format is <path.html> and the graph path in png format is <path.png>")`

    **Output Requirements:**
    - **analysis_report**: Focused markdown report for this analysis step including methodology, findings, and interpretation
    - **metrics**: List of all calculated values with descriptive labels
    - **image_html_path**: Path to HTML visualization (empty string if none created)
    - **image_png_path**: Path to PNG visualization (empty string if none created)
    - **conclusion**: Key insights and implications from this analysis step

    **Quality Standards:**
    - Use precise, professional language
    - Provide statistical context and confidence levels
    - Explain methodologies clearly
    - Include data validation steps
    - Reference specific numerical results
    - Use proper markdown formatting

    **Error Handling:**
    - If code fails, analyze the error and try alternative approaches
    - Document any data quality issues encountered
    - Validate results for statistical reasonableness
    - Provide fallback analysis methods when needed

    Execute this analysis step with precision and thoroughness, ensuring every insight is supported by quantitative evidence.
    provide the output in JSON format.
    """
    return prompt

async def run_execution_agent(ctx: RunContext[State]) -> str:
    """Execute the current step in the analysis workflow"""
    try:
        # Check if we have more steps to execute
        if ctx.deps.current_step_index >= len(ctx.deps.steps):
            return "All steps completed"
        
        # Execute current step
        response = await execution_agent.run(ctx.deps.user_query, deps=ctx.deps)
        response_data = response.output
        
        # Update state with results
        if response_data.image_html_path:
            ctx.deps.image_html_path.append(response_data.image_html_path)
        if response_data.image_png_path:
            ctx.deps.image_png_path.append(response_data.image_png_path)
        
        ctx.deps.analysis.append(response_data.analysis_report)
        ctx.deps.conclusion.append(response_data.conclusion)
        
        # Move to next step
        ctx.deps.current_step_index += 1
        
        # Prepare status message
        current_step = ctx.deps.steps[ctx.deps.current_step_index - 1]
        next_step = ctx.deps.steps[ctx.deps.current_step_index] if ctx.deps.current_step_index < len(ctx.deps.steps) else "Analysis complete"
        
        return f"Completed step: {current_step}\nResults: {response_data.analysis_report}\nMetrics: {response_data.metrics}\nConclusion: {response_data.conclusion}\nNext step: {next_step}"
        
    except Exception as e:
        print(f"Error in execution agent: {repr(e)}")
        return f"Error in execution agent: {repr(e)}"



# Supervisor Agent

class SupervisorAgentOutput(BaseModel):
    analysis_report: str = Field(description="The comprehensive analysis report in markdown format")
    conclusion: str = Field(description="The final conclusion of the analysis")

class FinalAnalysisOutput(BaseModel):
    analysis_report: str = Field(description="The comprehensive analysis report in markdown format")
    conclusion: str = Field(description="The final conclusion of the analysis")
    image_html_path: list[str] = Field(description="List of HTML visualization file paths", default_factory=list)
    image_png_path: list[str] = Field(description="List of PNG visualization file paths", default_factory=list)
    metrics: list[str] = Field(description="List of all analysis results and metrics", default_factory=list)


# Supervisor Agent
supervisor_agent = Agent(
    model=model,
    deps_type=State,
    result_type=SupervisorAgentOutput,
    tools=[Tool(run_execution_agent, takes_ctx=True)]
)

@supervisor_agent.system_prompt
async def get_supervisor_agent_system_prompt(ctx: RunContext[State]):
    
    if len(ctx.deps.analysis) > 0:
        prev_analysis_metrics = ctx.deps.analysis[-1]
    else:
        prev_analysis_metrics = ""
    prompt = f"""
    You are an analytical workflow supervisor responsible for orchestrating multi-step data analysis and synthesizing comprehensive reports.

    **Primary Responsibility:**
    Coordinate the execution of planned analysis steps and compile results into a unified, professional analytical report that directly addresses the user's query.

    **Available Resources:**
    - run_execution_agent: Execute individual analysis steps using specialized execution agents
      - Each execution agent has access to:
        - python_execution_tool: Statistical calculations and data processing
        - graph_generator: Interactive visualization creation (HTML/PNG output)

    **Current Analysis Context:**
    - User Query: {ctx.deps.user_query}
    - Dataset: {ctx.deps.file_name}
    - Planned Steps: {ctx.deps.steps}
    - Step Instructions: {ctx.deps.instructions}
    - Column Information: {ctx.deps.column_dict}
    - Current Step Index: {ctx.deps.current_step_index}
    - Previous Analysis: {prev_analysis_metrics}

    **Supervision Protocol:**
    
    1. **Query Assessment**
       - Analyze the user's request for key analytical objectives
       - Review planned steps for completeness and logical flow
       - Identify critical metrics and insights needed

    2. **Step-by-Step Execution**
    Note: No state carries between tool calls, please make sure to read the dataset in every step
       - Execute each analysis step sequentially using run_execution_agent
       - Monitor results for quality and relevance
       - Ensure each step contributes to answering the user's query
       - Track cumulative findings and metrics
       - Keep track of the steps executed and the steps remaining
       - End the execution when all the steps are executed

    3. **Results Integration**
       - Synthesize findings from all execution steps
       - Identify key patterns and insights across steps
       - Resolve any conflicting or unclear results
       - Compile comprehensive metrics list

    4. **Report Compilation**
       - Create unified analytical narrative
       - Ensure logical flow from methodology to conclusions
       - Include all relevant visualizations and metrics
       - Provide actionable recommendations

    **Execution Management:**
    - Call run_execution_agent to execute steps sequentially (state is automatically managed)
    - The execution agent will automatically move through steps using current_step_index
    - Monitor execution results and handle any errors or inconsistencies
    - Continue until all steps are completed

    **Final Output Specifications:**
    - **analysis_report**: Comprehensive markdown report including:
      * Executive Summary (2-3 key findings)
      * Methodology Overview
      * Step-by-step Analysis Results
      * Cross-step Insights and Patterns
      * Statistical Interpretations
      * Limitations and Assumptions
      * All visualizations and metrics from individual steps
    - **conclusion**: Strategic summary with actionable recommendations

    **Quality Assurance:**
    - Ensure all analysis steps are completed successfully
    - Verify that results directly address the user's query
    - Validate statistical interpretations for accuracy
    - Confirm visualizations support key findings
    - Maintain professional, data-driven narrative throughout

    **Success Criteria:**
    Deliver a cohesive analytical report that transforms raw data into actionable insights, directly answering the user's query with quantitative evidence and professional interpretation.
    provide the output in JSON format.
    """
    return prompt

# Graph Orchestration

@dataclass
class PlannerAgentNode(BaseNode[State]):
    """
    Planning the sections of the blog
    """
    async def run(self, ctx: GraphRunContext[State]) -> "SupervisorAgentNode":
        user_query = ctx.state.user_query
        file_name = ctx.state.file_name
        response = await planner_agent.run(user_query, deps=ctx.state)
        response_data = response.output
        ctx.state.steps = response_data.steps
        ctx.state.instructions = response_data.instructions
        ctx.state.column_dict = response_data.column_dict
        ctx.state.file_name = file_name
        # for debugging
        print(f'\n\n Steps: {ctx.state.steps}\n\n')
        print(f'\n\n Instructions: {ctx.state.instructions}\n\n')
        print(f'\n\n Column Dict: {ctx.state.column_dict}\n\n')
        print(f'\n\n File Name: {ctx.state.file_name}\n\n')
        return SupervisorAgentNode()

@dataclass
class SupervisorAgentNode(BaseNode[State]):
    """
    Getting media insights from the user query
    """
    async def run(self, ctx: GraphRunContext[State]) -> "End":
        user_query = ctx.state.user_query
        response = await supervisor_agent.run(user_query, deps=ctx.state)
        response_data = response.output
        
        # Create a comprehensive output BaseModel that includes state information
        final_output = FinalAnalysisOutput(
            analysis_report=response_data.analysis_report,
            conclusion=response_data.conclusion,
            image_html_path=ctx.state.image_html_path,
            image_png_path=ctx.state.image_png_path,
            metrics=ctx.state.analysis
        )

        return End(final_output)


# for running agent in streamlit app
def run_full_agent(user_query: str, file_name: str):

    state = State(user_query=user_query, file_name=file_name)
    graph = Graph(nodes=[PlannerAgentNode, SupervisorAgentNode])
    result = graph.run_sync(PlannerAgentNode(), state=state)
    result = result.output
    return result


# for running agent in terminal
async def run_full_agent_async(user_query: str, file_name: str):
    
    state = State(user_query=user_query, file_name=file_name)
    graph = Graph(nodes=[PlannerAgentNode, SupervisorAgentNode])
    result = await graph.run(PlannerAgentNode(), state=state)
    result = result.output
    return result

# for running agent in terminal
async def main():
    user_prompt = "analyze the data"
    file_name = "vgsales.csv"
    result = await run_full_agent_async(user_prompt, file_name)
    print('\n\n\n')
    print(result.analysis_report)
    print('\n\n\n')

if __name__ == "__main__":
    asyncio.run(main())