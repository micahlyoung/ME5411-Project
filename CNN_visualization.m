classdef CNN_visualization
    methods (Static)
        function stop = plotTrainingProgress(info)
            persistent plotHandles; 
            
            cmap = parula(10); 
            stop = false;

            if info.State == "start"
                figure('Units','normalized','Position',[0.2 0.2 0.6 0.5]);
                
                ax1 = subplot(1, 2, 1);
                
                plotHandles.trainAcc = animatedline('Color', cmap(2,:), 'Marker', '.', 'LineStyle', '-', 'LineWidth', 1.5);
                plotHandles.valAcc = animatedline('Color', cmap(8,:), 'Marker', '*', 'LineStyle', '-', 'LineWidth', 1.5);
                
                xlabel(ax1, "Iteration");
                ylabel(ax1, "Accuracy");
                title(ax1, 'Accuracy Progress');
                legend(ax1, 'Training', 'Validation');
                grid(ax1, 'on');

                ax2 = subplot(1, 2, 2);

                plotHandles.trainLoss = animatedline('Color', cmap(2,:), 'Marker', '.', 'LineStyle', '-', 'LineWidth', 1.5);
                plotHandles.valLoss = animatedline('Color', cmap(8,:), 'Marker', '*', 'LineStyle', '-', 'LineWidth', 1.5);

                xlabel(ax2, "Iteration");
                ylabel(ax2, "Loss");
                title(ax2, 'Loss Progress');
                legend(ax2, 'Training', 'Validation'); 
                grid(ax2, 'on');
            end

            if info.State == "iteration"
                if ~isempty(info.TrainingAccuracy)
                    addpoints(plotHandles.trainAcc, info.Iteration, info.TrainingAccuracy);
                end
                if ~isempty(info.ValidationAccuracy)
                    addpoints(plotHandles.valAcc, info.Iteration, info.ValidationAccuracy);
                end

                if ~isempty(info.TrainingLoss)
                    addpoints(plotHandles.trainLoss, info.Iteration, info.TrainingLoss);
                end

                if isfield(info, 'ValidationLoss') && ~isempty(info.ValidationLoss)
                    addpoints(plotHandles.valLoss, info.Iteration, info.ValidationLoss);
                end
                drawnow limitrate;
            end
        end
    end
end


