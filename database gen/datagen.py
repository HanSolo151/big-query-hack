import json
import csv
import random
import uuid
from datetime import datetime, timedelta

# Predefined values for synthetic generation
SOURCE_TYPES = ["log", "config", "doc", "incident"]
ENVIRONMENTS = ["dev", "staging", "prod"]
TOOLS = ["github-actions", "argo", "jenkins", "circleci"]
SEVERITIES = ["info", "warn", "error", "critical"]
CATEGORIES = ["Authentication", "Database", "Kubernetes", "Networking", "Storage", "Monitoring"]
PRIORITIES = ["Low", "Medium", "High", "Critical"]
SERVICES = ["user-service", "order-service", "payment-service", "inventory-service"]
CLUSTERS = ["cluster-a", "cluster-b", "cluster-c"]
NAMESPACES = ["auth", "payments", "orders", "inventory"]
TAGS = [["db", "timeout"], ["k8s", "crash"], ["oauth", "config"], ["ci", "pipeline"], ["network", "latency"]]

# Example text snippets
TITLES = [
    "Database connection timeout",
    "Auth config mismatch",
    "K8s Pod CrashLoopBackOff",
    "Pipeline build failure",
    "Service memory leak detected",
    "Network latency spike",
    "Storage volume out of space"
]

DESCRIPTIONS = [
    "Service failed to connect to DB due to high latency.",
    "Mismatch in OAuth redirect URL caused login failures.",
    "Pod repeatedly crashed due to missing environment variable.",
    "Build pipeline failed due to syntax error in config.",
    "Service restarted due to excessive memory usage.",
    "Observed high packet drop between nodes.",
    "Persistent volume reached 100% capacity."
]

CONFIG_SNIPPETS = [
    "DB_HOST=prod-db.company.com\nDB_PORT=5432",
    "oauth_redirect_url: https://app.company.com/callback",
    "replicas: 3\nimage: service:v2.1.0",
    "stages:\n  - build\n  - test\n  - deploy"
]

DOC_SNIPPETS = [
    "If login fails, check OAuth provider configuration.",
    "For DB issues, ensure correct host and port are set.",
    "Pod crashes may be due to missing environment variables.",
    "Pipeline failures often come from invalid YAML indentation."
]

RESOLUTIONS = [
    "Restarted DB pod and scaled replicas.",
    "Updated redirect URLs in config file.",
    "Added missing environment variable in deployment.",
    "Fixed YAML syntax in pipeline config.",
    "Increased memory limits for container.",
    "Reconfigured network policy.",
    "Extended storage volume by 50GB."
]

def random_datetime():
    start = datetime(2025, 1, 1)
    end = datetime(2025, 9, 1)
    return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))

def generate_record():
    source_type = random.choice(SOURCE_TYPES)
    created_at = random_datetime()
    updated_at = created_at + timedelta(minutes=random.randint(5, 300))
    return {
        "log_id": str(uuid.uuid4()),
        "title": random.choice(TITLES),
        "description": random.choice(DESCRIPTIONS),
        "source_type": source_type,
        "incident_id": str(uuid.uuid4()) if source_type == "incident" else None,
        "service": random.choice(SERVICES),
        "environment": random.choice(ENVIRONMENTS),
        "cluster": random.choice(CLUSTERS),
        "namespace": random.choice(NAMESPACES),
        "pod": f"{random.choice(SERVICES)}-{uuid.uuid4().hex[:5]}",
        "container": f"{random.choice(SERVICES)}-container",
        "file_path": "/configs/auth.yaml" if source_type == "config" else None,
        "commit_sha": uuid.uuid4().hex[:7],
        "tool": random.choice(TOOLS),
        "configs": random.choice(CONFIG_SNIPPETS) if source_type == "config" else None,
        "docs_faq": random.choice(DOC_SNIPPETS) if source_type == "doc" else None,
        "status": random.choice(["open", "in-progress", "resolved"]),
        "resolution": random.choice(RESOLUTIONS) if source_type in ["incident", "log"] else None,
        "severity": random.choice(SEVERITIES),
        "category": random.choice(CATEGORIES),
        "priority": random.choice(PRIORITIES),
        "tags": random.choice(TAGS),
        "created_at": created_at.isoformat(),
        "updated_at": updated_at.isoformat()
    }

def generate_dataset(n=1000, json_file="incident_dataset.json", csv_file="incident_dataset.csv"):
    records = [generate_record() for _ in range(n)]

    # Save as JSON
    with open(json_file, "w") as jf:
        json.dump(records, jf, indent=2)

    # Save as CSV
    with open(csv_file, "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    print(f"Generated {n} records -> {json_file}, {csv_file}")

if __name__ == "__main__":
    generate_dataset(5000)
