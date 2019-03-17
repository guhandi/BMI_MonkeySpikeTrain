function [ldaModel, C] = ClassifyAngle(data)

trial = data;
[n,d] = size(trial);
NUMTRIALS = n;
REACHING = d;
numNeurons = 98;
numGroups = 10;
x_spikeData = cell(numNeurons,1); %zeros(numNeurons,totalSamples); %total spikes for each neuron

startSpikes = zeros(numNeurons,1);

x_appendTimeSeries = zeros(numNeurons,1);
xpos = 0; ypos = 0;
y_appendTheta = 0;
samples=0;
start_window = 300;

maxx = 0; minx=0;
for angle=1:REACHING
    for trials=1:NUMTRIALS
        
        samples = samples+1;
        
        spikes = trial(trials,angle).spikes;
        x_appendTimeSeries = [x_appendTimeSeries spikes];
        startSpikes = [startSpikes spikes(:,1:start_window)]; %classification
        
        handpos = trial(trials,angle).handPos;
        x=handpos(1,:);
        y=handpos(2,:);
        z=handpos(3,:);
        theta = atan(y./x);
        
        y_appendTheta = [y_appendTheta theta];
        xpos = [xpos x];
        ypos = [ypos y];
        
        for i=1:numNeurons
            neuron = spikes(i,:);
            active = find(neuron == 1);
            dat = x_spikeData{i};
            newdat = [dat theta(active)];
            x_spikeData{i} = newdat;
            
        end
        
        %Group stuff
        
        if (max(x)>maxx)
            maxx =max(x);
        end
        if (min(x)<minx)
            minx = min(x);
        end
        
    end
end

rangex = maxx - minx;
%integerPositions = floor(rangex * rangey)


for i=1:numNeurons
    for j=1:numNeurons
        neuroni = x_appendTimeSeries(i,:);
        neuronj = x_appendTimeSeries(j,:);
        covar = xcorr(neuroni,neuronj,0,'coeff');
        C(i,j) = covar;
    end
    %autocorrelation = C(i,i) set to zero;
    C(i,i) = 0;
end

weightedActivity = C * x_appendTimeSeries;
startSpikes(:,1) = [];
weightStart = C*startSpikes;

angleLabels = ones(1,800);
angleLabels(101:200) = 2;
angleLabels(201:300) = 3;
angleLabels(301:400) = 4;
angleLabels(401:500) = 5;
angleLabels(501:600) = 6;
angleLabels(601:700) = 7;
angleLabels(701:800) = 8;

tr=0;
for a=1:start_window:length(weightStart)
    tr=tr+1;
    startData = weightStart(:, a : a+start_window-1);
    meanActivity(tr,:) = (mean(startData,2))'; %average activity for each neruon within the first 300ms
    
end

%randomize order of dataset
[row, col] = size(meanActivity);
idx = randperm(row);
input = meanActivity(idx,:);
label = angleLabels(idx);

%split to test and train
num_training = 700;
trainData = input(1:num_training, :);
trainLabel = label(1:num_training);
testData = input(num_training+1:end, :);
testLabel = label(num_training+1 : end);

%train & test
ldaModel = fitcdiscr(trainData,trainLabel);
class1 = predict(ldaModel,testData)';

total = length(class1);
true = find(class1 == testLabel)
correct = length(true);
accuracy = correct / total

end





