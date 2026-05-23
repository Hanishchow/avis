from birdnet_integration import BirdNETProcessor
proc = BirdNETProcessor(min_conf=0.01)
detections = proc.analyze_file("test_bird_cardinal.wav")
print(f"Detections: {len(detections)}")
for d in detections[:5]:
    print(f"  {d['common_name']}: {d['confidence']*100:.1f}%")
