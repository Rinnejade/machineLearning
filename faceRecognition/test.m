load X.mat
load y.mat

trainPercent = 60;
crossValidatePercent = 80;
testPercent = 100;

trainNum = ceil(size(X,1)*trainPercent/100);
cvNum = ceil(size(X,1)*crossValidatePercent/100);
testX = X(cvNum:end,:);
testy = y(cvNum:end,:);

load Theta1.mat
load Theta2.mat
pred = predict(Theta1,Theta2,X);
fprintf('\nComplete Set Accuracy: %f\n', mean(double(pred == y)) * 100);
pred = predict(Theta1,Theta2,testX);
fprintf('\nTesting Set Accuracy: %f\n', mean(double(pred == testy)) * 100);


