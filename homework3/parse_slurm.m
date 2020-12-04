function [] = parse_slurm()
    fileID = fopen('results.txt','r');
    line = fgetl(fileID)
    scores = [];
    results = {};
    
    disp("Reading file...")
    while line ~= -1
        episode = str2double(extractBetween(line, "Ep: ",","));
        if isempty(episode) == 1
            line = fgetl(fileID)
            continue;
        elseif episode == 0
            results = [results, scores];
            scores = [];
        end
        score = str2double(sprintf("%.2f", str2double(extractBetween(line, "Score: ",","))));
        scores = [scores, [score]];
        line = fgetl(fileID);
    end
    results = [results, scores];
    fclose('all')

    for k=1:length(results)
        figure
        result=results{k};
        plot(1:length(result),result, 'ro');
        title("Hyperparameter Set " + k)
        xlabel("Episode")
        ylabel("Q Value")
        xlim([0 1000])
        ylim([0 500000000])
    end
end