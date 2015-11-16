function InputOutput
train1 = load('dataset1-a8a-training.txt','rt');
test1 = load('dataset1-a8a-testing.txt','rt');
train2 = load('dataset1-a9a-training.txt','rt');
test2 = load('dataset1-a9a-testing.txt','rt');
b = 0;
lamda1 = 10^-4;
lamda2 = 5*10^-5;

%[x1h, y1h] = PegasosAlgorithmhinge(train1, lamda1, test1);
%[x2h, y2h] = PegasosAlgorithmhinge(train2, lamda2, test2);
[x1l, y1l] = PegasosAlgorithmlog(train1, lamda1, test1);
%[xl2, y2l] = PegasosAlgorithmlog(train2, lamda2, test2);
labelh1 = 'hinge-loss test1';
labelh2 = 'hinge-loss test2';
labell1 = 'log-loss test1';
labell2 = 'log-loss test2';
Output(x1l, y1l, labell1);

end


function [resultX, resultY] = PegasosAlgorithmhinge(S, lamda, test)
resultX = zeros(1,10);
resultY = zeros(1,10);
step = 0.1;
[m, n] = size(S);
T = 5*m;
Wout = zeros(1,n-1);
for t = 1:T
    i = randperm(m, 1);
    eta = 1/(lamda*t);
    if S(i,n) < 1
        Wout = (1 - eta*lamda)*Wout + eta*S(i, n)*S(i,1:n-1);
    else
        Wout = (1 - eta*lamda)*Wout;
    end
    if t == round(step*T)
        resultX(round(step*10)) = step;
        resultY(round(step*10)) = TestPegasos(Wout, test);
        step = step + 0.1;
    end
end   
end

function [resultX, resultY] = PegasosAlgorithmlog(S, lamda, test)
resultX = zeros(1,10);
resultY = zeros(1,10);
step = 0.1;
[m, n] = size(S);
T = 5*m;
Wout = zeros(1,n-1);
for t = 1:T
    i = randperm(m, 1);
    eta = 1/(lamda*t);
    yi = S(i,n);
    Z = sum(Wout.*S(i,1:n-1));
    Wout = Wout - eta*-(yi/(1+exp(yi*Z))*S(i,1:n-1));
    if t == round(step*T)
        resultX(round(step*10)) = step;
        resultY(round(step*10)) = TestPegasos(Wout, test);
        step = step + 0.1;
    end
end   
end

function testerror = TestPegasos(W, test)
[m, n] = size(test);
testerror = 0.0;
myresult = zeros(1, m);
for i = 1:m
    newvec = W.*test(i,1:n-1);
    value = sum(newvec);
    if value < 0
        myresult(i) = -1;
    else
        myresult(i) = 1;
    end
    if myresult(i) ~= test(i,end)
        testerror = testerror + 1.0;
    end
end
testerror = testerror / m;
end

function Output(x, y, label)
plot(x, y);
set(gca,'ytick',0:0.1:1);
axis([0 1 0 1]);
title(label);
ylabel('test error');
xlabel('T');
end

