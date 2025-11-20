from log_config import setup_logging
from agents.code_generator import generate_spark_job
from agents.validator import validate_generated_code

logger = setup_logging()
logger.info("Application Started...")

if __name__ == "__main__":
    instruction = "Read a CSV file and aggregate users by department and save the result to a text file."
    logger.info(f"Instruction for LLM: {instruction}")
    result = generate_spark_job(instruction)

    logger.info(f"Running Validation checks on the generated code..")
    validate_generated_code(result)
