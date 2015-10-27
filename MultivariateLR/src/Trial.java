/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author rj
 */
public class Trial {
    public static void main(String[] args) {
        int A=2,B=5,Y=0;
        int i=B,j=0;
        
        do{

            j=Y;
            Y=0;
            do{
                Y = Y + A;
                j--;
            }while(j>0);
            i--;
            
            System.out.println("Y:   " + Y );
        }while(i>0);
        System.out.println("Y: " + Y);
        
        
    }
}
