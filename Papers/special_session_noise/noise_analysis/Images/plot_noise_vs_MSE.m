clear
close all

% Define size of figure to print
figSize=[700 500];

V = 0:59;
N = numel(V);
C = cell(1,N);
disp(N);
for k = 1:N
    F = sprintf('./SNN_DPE_data/MG_noise_vs_MSE_synapse_%d.csv',V(k));
    C{k} = csvread(F,0,0);
end
disp(C{1});

for i = 1:N
    file = sprintf('./SNN_DPE_data/MG_noise_vs_MSE_synapse_%d.csv',V(i));
    m = csvread(file,0,0);
    if i==2
        %m
    end
    for r = 1:size(m,1)
        if all(isnan(m(r,:)))
            m= m(1:r-1,2:end);
            break;
        end
    end
   
    for r = 2:size(m,2)
        if all(isnan(m(:,r)))
            m= m(1:end,1:r-1);
            break;
        end
    end
    if all(isnan(m(:,1)))
        m= m(:,2:end);
    end
    %m
    if i==1
        mp = zeros(size(m,1), size(m,2), N);
    end
    mp(:,:,i) = m;
    
end

%  average
plot_data_synapse=mean(mp,3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% dataset
V = 0:59;
N = numel(V);
C = cell(1,N);
disp(N);
for k = 1:N
    F = sprintf('./SNN_DPE_data/MG_noise_vs_MSE_dataset_%d.csv',V(k));
    C{k} = csvread(F,0,0);
end
disp(C{1});

for i = 1:N
    file = sprintf('./SNN_DPE_data/MG_noise_vs_MSE_dataset_%d.csv',V(i));
    m = csvread(file,0,0);
    if i==2
        %m
    end
    for r = 1:size(m,1)
        if all(isnan(m(r,:)))
            m= m(1:r-1,2:end);
            break;
        end
    end
   
    for r = 2:size(m,2)
        if all(isnan(m(:,r)))
            m= m(1:end,1:r-1);
            break;
        end
    end
    if all(isnan(m(:,1)))
        m= m(:,2:end);
    end
    %m
    if i==1
        mp = zeros(size(m,1), size(m,2), N);
    end
    mp(:,:,i) = m;
    
end

%  average
plot_data_dataset=mean(mp,3);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% write_noise
V = 0:39;
N = numel(V);
C = cell(1,N);
disp(N);
for k = 1:N
    F = sprintf('./SNN_DPE_data/MG_noise_vs_MSE_write_noise_%d.csv',V(k));
    C{k} = csvread(F,0,0);
end
disp(C{1});

for i = 1:N
    file = sprintf('./SNN_DPE_data/MG_noise_vs_MSE_write_noise_%d.csv',V(i));
    m = csvread(file,0,0);
    if i==2
        %m
    end
    for r = 1:size(m,1)
        if all(isnan(m(r,:)))
            m= m(1:r-1,2:end);
            break;
        end
    end
   
    for r = 2:size(m,2)
        if all(isnan(m(:,r)))
            m= m(1:end,1:r-1);
            break;
        end
    end
    if all(isnan(m(:,1)))
        m= m(:,2:end);
    end
    %m
    if i==1
        mp = zeros(size(m,1), size(m,2), N);
    end
    mp(:,:,i) = m;
    
end

%  average
plot_data_write_noise=mean(mp,3);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% all_noise
V = 0:59;
N = numel(V);
C = cell(1,N);
disp(N);
for k = 1:N
    F = sprintf('./SNN_DPE_data/MG_noise_vs_MSE_all_noise_%d.csv',V(k));
    C{k} = csvread(F,0,0);
end
disp(C{1});

for i = 1:N
    file = sprintf('./SNN_DPE_data/MG_noise_vs_MSE_all_noise_%d.csv',V(i));
    m = csvread(file,0,0);
    if i==2
        %m
    end
    for r = 1:size(m,1)
        if all(isnan(m(r,:)))
            m= m(1:r-1,2:end);
            break;
        end
    end
   
    for r = 2:size(m,2)
        if all(isnan(m(:,r)))
            m= m(1:end,1:r-1);
            break;
        end
    end
    if all(isnan(m(:,1)))
        m= m(:,2:end);
    end
    %m
    if i==1
        mp = zeros(size(m,1), size(m,2), N);
    end
    mp(:,:,i) = m;
    
end

%  average
plot_data_all_noise=mean(mp,3);

% plot
figure()
hold on
grid on

plot((plot_data_synapse(:,3)),(plot_data_synapse(:,1)),'-','Color','[0.1290 0.7940 0.6250]','LineWidth',3); 
plot((plot_data_synapse(:,3)),(plot_data_synapse(:,2)),'--','Color','[0.2940 0.8840 0.5560]','LineWidth',3); 
plot((plot_data_synapse(:,3)),(plot_data_dataset(:,1)),'-','Color','[0.3290 0.9940 0.4250]','LineWidth',3); 
plot((plot_data_synapse(:,3)),(plot_data_dataset(:,2)),'--','Color','[0.4290 0.9740 0.3250]','LineWidth',3); 
plot((plot_data_synapse(:,3)),(plot_data_write_noise(:,1)),'-','Color','[0.5290 0.1940 0.2250]','LineWidth',3); 
plot((plot_data_synapse(:,3)),(plot_data_write_noise(:,2)),'--','Color','[0.6290 0.2040 0.1250]','LineWidth',3); 
plot((plot_data_synapse(:,3)),(plot_data_all_noise(:,1)),'-','Color','[0.7290 0.3940 0.9250]','LineWidth',3); 
plot((plot_data_synapse(:,3)),(plot_data_all_noise(:,2)),'--','Color','[0.8290 0.4040 0.8250]','LineWidth',3); 
%xticks([0 0.2 0.4 0.6 0.8 1.0])
set(gca,'fontsize',18)
ylabel('Normalized Root Mean Square Error','Interpreter','latex','FontSize', 18)
xlabel('Synapse Noise(std. dev.)','Interpreter','latex','FontSize', 18)
legend({'Training (synapse noise)','Testing (synapse noise)','Training (data noise)','Testing (data noise)','Training (write noise)','Testing (write noise)','Training (all noise)','Testing (all noise)'}, ...
        'Location','northwest', ...
        'Interpreter','latex','FontSize',12)
title('Mackey-Glass time-series','Interpreter','latex','FontSize', 18)
set(gcf,'Position',[100 100 figSize])
print('Filter','-dpng','-r300')
