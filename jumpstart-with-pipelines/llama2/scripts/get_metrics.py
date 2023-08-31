import os
os.system('python3 -m pip install -U sagemaker datasets')

import json
import argparse
import boto3
from sagemaker.session import Session
from sagemaker.analytics import TrainingJobAnalytics


session = Session(boto3.session.Session(region_name="us-east-1"))


def read_parameters():
    """
    Read job parameters
    Returns:
        (Namespace): read parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_job_name', type=str, default="")
    parser.add_argument('--metric_names', type=str, default="")
    parser.add_argument('--output_path', type=str, default="/opt/ml/processing/evaluation")
    params, _ = parser.parse_known_args()
    return params


if __name__ == "__main__":
    
    # reading job parameters
    args = read_parameters()
    
    # Create a TrainingJobAnalytics object
    analytics = TrainingJobAnalytics(
        training_job_name=args.training_job_name, 
        metric_names=[
            metric.rstrip() for metric in args.metric_names.split(',')
        ],
        sagemaker_session=session
    )
    
    print(analytics.dataframe())
    
    report_dict = {"metrics": {}}
    for metric in analytics.dataframe().to_dict(orient='records'):
        report_dict["metrics"][
            metric["metric_name"].split(":")[-1].replace("-", "_")
        ] = {"value": metric["value"]}

    output_dir = args.output_path
    os.makedirs(output_dir, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
    
    print("Done")
        
