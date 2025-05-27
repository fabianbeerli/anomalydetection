#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <string>
using namespace std;

// Compute L2 distance between two vectors
double l2_distance(const vector<double>& a, const vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    return sqrt(sum);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <input_csv_file>" << endl;
        return 1;
    }
    string input_file = argv[1];
    ifstream file(input_file);
    if (!file.is_open()) {
        cerr << "Error: Could not open input file " << input_file << endl;
        return 1;
    }

    // Read CSV into matrix (each row = stock, columns = features)
    vector<vector<double>> stocks;
    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        string cell;
        vector<double> row;
        while (getline(ss, cell, ',')) {
            row.push_back(stod(cell));
        }
        stocks.push_back(row);
    }
    file.close();

    size_t n_stocks = stocks.size();
    if (n_stocks == 0) {
        cerr << "No data found in file." << endl;
        return 1;
    }

    // Compute anomaly scores: average L2 distance to all other stocks
    vector<double> scores(n_stocks, 0.0);
    for (size_t i = 0; i < n_stocks; ++i) {
        double sum_dist = 0.0;
        for (size_t j = 0; j < n_stocks; ++j) {
            if (i == j) continue;
            sum_dist += l2_distance(stocks[i], stocks[j]);
        }
        scores[i] = sum_dist / (n_stocks - 1);
    }

    // Output scores
    ofstream fres(string(input_file) + "_AIDA_scores.dat");
    fres << n_stocks << endl;
    for (size_t i = 0; i < scores.size(); ++i)
        fres << scores[i] << endl;
    fres.close();

    // Compute mean and stddev
    double mean = 0.0, stddev = 0.0;
    for (double s : scores) mean += s;
    mean /= scores.size();
    for (double s : scores) stddev += (s - mean) * (s - mean);
    stddev = sqrt(stddev / scores.size());
    double threshold = mean + 1.5 * stddev;
    cout << "mean: " << mean << ", stddev: " << stddev << ", threshold: " << threshold << endl;

    // Output anomalies
    ofstream fanom(string(input_file) + "_AIDA_anomalies.csv");
    fanom << "index,stock_idx,score" << endl;
    for (size_t i = 0; i < scores.size(); ++i) {
        if (scores[i] > threshold)
            fanom << i << "," << i << "," << scores[i] << endl;
    }
    fanom.close();

    cout << "Done. Scores and anomalies written." << endl;
    return 0;
}