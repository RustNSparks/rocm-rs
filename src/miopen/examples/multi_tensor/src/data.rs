use std::collections::BTreeSet;


pub fn prepare_data() -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<String>) {
    let data = read_iris();

    let x_data = data
        .iter()
        .map(|(features, _)| features.clone())
        .collect::<Vec<Vec<f32>>>();

    let y_data = data
        .iter()
        .map(|(_, label)| label.clone())
        .collect::<Vec<String>>();

    let (y_target, class_labels) = one_hot_encode(&y_data);

    (x_data, y_target, class_labels)
}

fn read_iris() -> Vec<(Vec<f32>, String)> {
    let mut rdr = csv::Reader::from_path("src/iris.csv").expect("Cannot open iris.csv");
    let mut data = Vec::new();

    for result in rdr.records() {
        let record = result.expect("Error reading record");
        let features: Vec<f32> = record
            .iter()
            .take(4)
            .map(|s| s.parse::<f32>().expect("Error parsing feature"))
            .collect();
        let label = record.get(4).unwrap().to_string();
        data.push((features, label));
    }

    data
}

fn one_hot_encode(labels: &[String]) -> (Vec<Vec<f32>>, Vec<String>) {
    let unique_labels: Vec<String> = labels
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();

    let mut one_hot = Vec::with_capacity(labels.len());

    for label in labels {
        let mut encoding = vec![0.0; unique_labels.len()];
        if let Some(pos) = unique_labels.iter().position(|l| l == label) {
            encoding[pos] = 1.0;
        }
        one_hot.push(encoding);
    }

    (one_hot, unique_labels)
}
