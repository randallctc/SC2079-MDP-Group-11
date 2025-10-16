import torch
torch.hub._validate_not_a_forked_repo = lambda a,b,c: True
torch.hub.set_dir('./torch_cache')
torch.hub.load(
    r"C:\users\randa\yolov5",
    'custom',
    path=r"C:\Users\randa\OneDrive\Documents\GitHub\SC2079-MDP-Group-11\best.pt",
    source='local',
    force_reload=True 
)