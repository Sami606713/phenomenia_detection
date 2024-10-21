import os 
from pathlib import Path

# make folders
dir_=[
    os.path.join("data","raw"),
    os.path.join("data","processed"),
    "Notebooks",
    "Models",
    "reports",
    "src",
    os.path.join("src","data"),
    os.path.join("src","models"),

    "Config"
]

for folder in dir_:
    if not os.path.exists(folder):
        os.makedirs(folder)
        # add the .git keep file
        git_keep = os.path.join(folder,".gitkeep")

        with open(git_keep,"w") as f:
            pass
    else:
        print(f"{folder} already exists")

# place files in each folder
files=[
    os.path.join("src","__init__.py"),
    os.path.join("src","utils.py"),
    os.path.join('src','data',"__init__.py"),
    os.path.join('src','data',"data_loader.py"),
    os.path.join('src','data',"data_transformation.py"),

    os.path.join('src','models',"__init__.py"),
    os.path.join('src','models',"train_model.py"),
    os.path.join('src','models',"predict_model.py"),

    os.path.join('Config',"config.yml"),
    
    "README.md",
    "requirements.txt",
    ".gitignore",
    "LICENSE",
    "setup.py",
    "Dockerfile",
    ".dockerignore",
    "test_environment.py",
    "dvc.yaml",
    "get_data.py",
    "app.py",
    ".env"
]

for file in files:
    try:
        if not os.path.exists(file) or os.path.getsize(file)==0:
            with open(file,"w") as f:
                pass
    except Exception as e:
        print(f"Error creating {file}: {str(e)}")