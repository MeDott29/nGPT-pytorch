import React, { useState, useRef, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { AlertDialog, AlertDialogContent, AlertDialogDescription, AlertDialogHeader, AlertDialogTitle } from '@/components/ui/alert-dialog';
import { Grid, RefreshCw, Download, Trash2, Play, Square } from 'lucide-react';

const MNISTCrayon = () => {
    const [isDrawing, setIsDrawing] = useState(false);
    const [images, setImages] = useState([]);
    const [isSimulating, setIsSimulating] = useState(false);
    const [progress, setProgress] = useState(0);
    const [showAlert, setShowAlert] = useState(false);
    const canvasRef = useRef(null);
    const gridCanvasRef = useRef(null);
    const animationRef = useRef(null);
    const CANVAS_SIZE = 280;
    const GRID_SIZE = 1080;
    const CELL_SIZE = 32;
    const GRID_CELLS = 34;
    const SIMULATION_DIGITS = [
        "M140,100 C140,50 180,50 180,100 C180,150 140,150 140,100",  // 0
        "M160,50 L160,150",  // 1
        "M140,70 C180,50 180,100 140,130 L180,150",  // 2
        "M140,50 C180,50 180,100 140,100 C180,100 180,150 140,150",  // 3
        "M180,150 L180,50 L140,100 L180,100",  // 4
        "M180,50 L140,50 L140,100 C180,100 180,150 140,150",  // 5
        "M180,50 L140,50 L140,150 C180,150 180,100 140,100",  // 6
        "M140,50 L180,50 L160,150",  // 7
        "M160,100 C140,50 180,50 160,100 C140,150 180,150 160,100",  // 8
        "M180,100 C180,50 140,50 140,100 L180,150"  // 9
    ];

    useEffect(() => {
        clearCanvas();
        clearGrid();
    }, []);

    const startSimulation = async () => {
        setIsSimulating(true);
        setProgress(0);
        clearGrid();
        let currentImages = [];

        for (let i = 0; i < SIMULATION_DIGITS.length; i++) {
            clearCanvas();
            await new Promise(resolve => setTimeout(resolve, 500));
            
            const canvas = canvasRef.current;
            const ctx = canvas.getContext('2d');
            ctx.strokeStyle = 'black';
            ctx.lineWidth = 20;
            ctx.lineCap = 'round';
            
            const path = new Path2D(SIMULATION_DIGITS[i]);
            
            // Animated drawing effect
            let length = 0;
            const draw = () => {
                ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
                ctx.fillStyle = 'white';
                ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
                
                ctx.save();
                ctx.setLineDash([length, 1000]);
                ctx.stroke(path);
                ctx.restore();
                
                length += 10;
                if (length < 200) {
                    animationRef.current = requestAnimationFrame(draw);
                } else {
                    const scaledCanvas = document.createElement('canvas');
                    scaledCanvas.width = CELL_SIZE;
                    scaledCanvas.height = CELL_SIZE;
                    const scaledCtx = scaledCanvas.getContext('2d');
                    scaledCtx.fillStyle = 'white';
                    scaledCtx.fillRect(0, 0, CELL_SIZE, CELL_SIZE);
                    scaledCtx.drawImage(canvas, 0, 0, CELL_SIZE, CELL_SIZE);
                    
                    currentImages = [...currentImages, scaledCanvas.toDataURL()];
                    setImages(currentImages);
                    updateGrid(currentImages);
                    setProgress((i + 1) * 10);
                    
                    if (i === SIMULATION_DIGITS.length - 1) {
                        setShowAlert(true);
                        setIsSimulating(false);
                    }
                    resolve();
                }
            };
            draw();
            await new Promise(resolve => setTimeout(resolve, 1000));
        }
    };

    const clearCanvas = () => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    };

    const clearGrid = () => {
        setImages([]);
        const gridCanvas = gridCanvasRef.current;
        const ctx = gridCanvas.getContext('2d');
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, GRID_SIZE, GRID_SIZE);
    };

    const updateGrid = (imageList) => {
        const gridCanvas = gridCanvasRef.current;
        const ctx = gridCanvas.getContext('2d');
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, GRID_SIZE, GRID_SIZE);

        imageList.forEach((dataUrl, i) => {
            const img = new Image();
            img.onload = () => {
                const x = (i % GRID_CELLS) * (GRID_SIZE / GRID_CELLS);
                const y = Math.floor(i / GRID_CELLS) * (GRID_SIZE / GRID_CELLS);
                ctx.drawImage(img, x, y, GRID_SIZE/GRID_CELLS, GRID_SIZE/GRID_CELLS);
            };
            img.src = dataUrl;
        });
    };

    return (
        <div className="max-w-6xl mx-auto p-6 space-y-6">
            <Card>
                <CardHeader>
                    <CardTitle className="flex justify-between items-center">
                        <span>MNIST Crayon - Automated Test</span>
                        <div className="space-x-2">
                            {!isSimulating ? (
                                <Button onClick={startSimulation} size="sm" className="bg-green-600 hover:bg-green-700">
                                    <Play className="w-4 h-4 mr-2" />
                                    Run Test
                                </Button>
                            ) : (
                                <Button disabled size="sm" className="bg-red-600">
                                    <Square className="w-4 h-4 mr-2" />
                                    Testing... {progress}%
                                </Button>
                            )}
                        </div>
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    <canvas
                        ref={canvasRef}
                        width={CANVAS_SIZE}
                        height={CANVAS_SIZE}
                        className="border border-gray-300 rounded-lg bg-white mx-auto"
                    />
                </CardContent>
            </Card>

            <Card>
                <CardHeader>
                    <CardTitle className="flex justify-between items-center">
                        <span>Generated Test Grid ({images.length}/10 digits)</span>
                        <div className="space-x-2">
                            <Button onClick={clearGrid} variant="outline" size="sm">
                                <Trash2 className="w-4 h-4" />
                            </Button>
                        </div>
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="border border-gray-300 rounded-lg overflow-hidden">
                        <canvas
                            ref={gridCanvasRef}
                            width={GRID_SIZE}
                            height={GRID_SIZE}
                            className="w-full h-auto bg-white"
                        />
                    </div>
                </CardContent>
            </Card>

            <AlertDialog open={showAlert} onOpenChange={setShowAlert}>
                <AlertDialogContent>
                    <AlertDialogHeader>
                        <AlertDialogTitle>Test Complete! 🎉</AlertDialogTitle>
                        <AlertDialogDescription>
                            Successfully generated all 10 digits (0-9) using SVG path animations.
                            Each digit was automatically drawn, processed, and added to the grid.
                        </AlertDialogDescription>
                    </AlertDialogHeader>
                </AlertDialogContent>
            </AlertDialog>
        </div>
    );
};

export default MNISTCrayon;