import pandas as pd
import toloka.client as toloka

def upload_task_to_toloka(token: str, pool_id: str, df: pd.DataFrame, overlap: int = 1, target: str = "SANDBOX") -> toloka.operations.Operation:
    toloka_client = toloka.TolokaClient(token, target)

    training_tasks = [toloka.task.Task(input_values=row, pool_id=pool_id) for row in df.to_dict(orient="rows")]

    task_suites = [
        toloka.task_suite.TaskSuite(
            pool_id=pool_id,
            overlap=overlap,
            tasks=[task]
        )
        for task in training_tasks
    ]
    task_suites_op = toloka_client.create_task_suites_async(task_suites)
    result = toloka_client.wait_operation(task_suites_op)
    return result
