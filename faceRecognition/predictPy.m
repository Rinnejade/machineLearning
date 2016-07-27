function p = predictPy(inImage)
% X = double(inImage);
X = double(imread(inImage)(:)');
load Theta1.mat
load Theta2.mat
m = size(X, 1);
num_labels = size(Theta2, 1);
p = zeros(size(X, 1), 1);
h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
[dummy, p] = max(h2, [], 2);
end
