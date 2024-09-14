package githubcode.slimcode;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseProblemException;
import com.github.javaparser.ast.CompilationUnit;
//import SpanContent;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.*;




public class SlimCode {

    public static void markFlag(int[] codeFlag,SpanContent spanContent,int flag,String code,boolean[] otherFlag){
        int startWord = spanContent.startWord;
        int endWord = spanContent.endWord;
        for (int i=startWord;i<=endWord-1;i++){
            codeFlag[i] = flag;
            if (otherFlag!=null){
                otherFlag[i] = true;
            }

        }
    }


//    public static int targetLength = 185;
    public static double ratio = 0.7;


    public static ArrayList<Integer> getRemovedIndex(String[] codeSplits,int[] codeFlag,List<String> allTokens,Map<Integer, List<String>> wordToSubtokenMap,int indexOfNoSig){
        ArrayList<Integer> removeIndex = new ArrayList<>();
        int sourceLen=allTokens.size();
        int targetLengthThis = (int) (sourceLen * ratio);
        int removeTargetLength = sourceLen - targetLengthThis;
        int removedLength=0;
        for (int i=8;i>0;i--){
            for (int j = codeSplits.length-1;j>=0;j--){
                if (codeFlag[j] == i){
                    removedLength +=  wordToSubtokenMap.get(j+indexOfNoSig).size();
                    if (removedLength > removeTargetLength){
                        return removeIndex;
                    }
                    removeIndex.add(j);
                }
            }
        }
        return removeIndex;

    }


    public static int id = 0;

    public static FileOutputStream fileOutputStream_log = null;
    public static OutputStreamWriter outputStreamWriter_log = null;
    public static BufferedWriter bufferedWriter_log = null;

    public static long astTime = 0;
    public static long labelTime = 0;
    public static long removeTime = 0;

    public static String removeCode(String code, Map map,List<String> allTokens,Map<Integer, List<String>> wordToSubtokenMap){

        if(fileOutputStream_log == null){
            try {
                fileOutputStream_log = new FileOutputStream("log.txt");
                outputStreamWriter_log = new OutputStreamWriter(fileOutputStream_log);
                bufferedWriter_log = new BufferedWriter(outputStreamWriter_log);
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }

        }


        long startTime = System.currentTimeMillis();
        ArrayList<SpanContent> identifierList = (ArrayList<SpanContent>) map.get("identifiers");
        ArrayList<SpanContent> invocationList = (ArrayList<SpanContent>) map.get("function_invocation");
        ArrayList<SpanContent> structureList = (ArrayList<SpanContent>) map.get("function_structure");
        ArrayList<SpanContent> signatureList = (ArrayList<SpanContent>) map.get("method_signature");
        ArrayList<SpanContent> simpleSymbolList = new ArrayList<SpanContent>();
        ArrayList<SpanContent> otherList = new ArrayList<SpanContent>();


        String[] codeSplits = code.split(" +");
        boolean[] structureFlag = new boolean[codeSplits.length];
        boolean[] invocationFlag = new boolean[codeSplits.length];
        boolean[] identifierFlag = new boolean[codeSplits.length];
        boolean[] simpleSymbolFlag = new boolean[codeSplits.length];

        int[] codeFlag = new int[codeSplits.length];
        // signature > identifier > structure > invocation > simple symbols
        for (SpanContent spanContent : signatureList){
            markFlag(codeFlag,spanContent,1,code,null);
        }
        for (SpanContent spanContent : identifierList){
            markFlag(codeFlag,spanContent,-1,code,identifierFlag);
        }
        for (SpanContent spanContent : structureList){
            markFlag(codeFlag,spanContent,-1,code,structureFlag);
        }
        for (SpanContent spanContent : invocationList){
            markFlag(codeFlag,spanContent,-1,code,invocationFlag);
        }

        // simple
        String[] simpleStr = new String[]{"=", "+", "-", "*", "/", "%", "!", ">",  "<", "|", "?", ":", "~", "&", "^", "(",
                "{", ")", "}", "[", ".", "]", ";", "\"", ",","==","++","--","!=",">=","<=","&&","||","<<",">>",">>>","\'"
        };
        List<String> simpleList = Arrays.asList(simpleStr);
        for(int i = 0;i< codeSplits.length;i++){
            if (simpleList.contains(codeSplits[i])){
                codeFlag[i] = 8;
                simpleSymbolFlag[i] =true;
            }
        }

        //other
        for (int i = 0; i<codeSplits.length;i++){
            if (codeFlag[i] == 0){
                int start = i;
                while (start < codeFlag.length && codeFlag[start] == 0){
                    start ++;
                }
                int end = start;
                for (int k=i;k<end;k++){
                    codeFlag[k] = 7;
                }
            }
        }

        for (int i = 0; i< codeSplits.length;i++){
            if (!simpleSymbolFlag[i] && structureFlag[i] && !identifierFlag[i]){
                codeFlag[i] = 6;
            }else if (!simpleSymbolFlag[i] && invocationFlag[i] && !identifierFlag[i]){
                codeFlag[i] = 5;
            }else if (!simpleSymbolFlag[i] && structureFlag[i] && identifierFlag[i]){
                codeFlag[i] = 2;
            }else if (!simpleSymbolFlag[i] && invocationFlag[i] && identifierFlag[i]){
                codeFlag[i] = 3;
            }else if (!simpleSymbolFlag[i] && !invocationFlag[i] && !structureFlag[i] && identifierFlag[i]){
                codeFlag[i] = 4;
            }
        }

        long endTime = System.currentTimeMillis();
	    long time = endTime - startTime;
	    labelTime += time;
        startTime = System.currentTimeMillis();
        String removedCode = "";


        ArrayList<Integer> removedIndex = getRemovedIndex(codeSplits, codeFlag,allTokens,wordToSubtokenMap,0);
        for (int index : removedIndex){
            removedCode += codeSplits[index] + " ";
            codeSplits[index] = "";
        }
//        System.out.println(removedCode);

        String new_code = String.join(" ",codeSplits);

       endTime = System.currentTimeMillis();
       time = endTime - startTime;
       removeTime += time;

        try {
            if (id < 1000){
                bufferedWriter_log.write(id + "：" + removedCode + "\n");
                bufferedWriter_log.write(id + "：" + code+"\n");
                bufferedWriter_log.write(id + "：" + new_code + "\n\n");
            }


        } catch (IOException e) {
            e.printStackTrace();
        }
        return new_code;
    }

    public static String remove(String code,List<String> allTokens,Map<Integer, List<String>> wordToSubtokenMap){

	    long startTime = System.currentTimeMillis();
        MyVisitor myVisitor = new MyVisitor(code);
        CompilationUnit cu = JavaParser.parse(myVisitor.code);
        myVisitor.visit(cu, null);
//        System.out.println(myVisitor.map);
        long endTime = System.currentTimeMillis();
       	long time = endTime - startTime;
	    astTime += time;

	    String removedCode = removeCode(code, myVisitor.map,allTokens,wordToSubtokenMap);
//        System.out.println(removedCode);
        return removedCode;
    }

    public static List<String> readFile(String path) throws IOException {
        FileInputStream fileInputStream = new FileInputStream(path);
        InputStreamReader inputStreamReader = new InputStreamReader(fileInputStream);
        BufferedReader bufferedReader = new BufferedReader(inputStreamReader);
        List<String> stringsList = new ArrayList<String>();
        String stringTmp = "";
        while ((stringTmp = bufferedReader.readLine())!=null){
            stringsList.add(stringTmp);
        }
        inputStreamReader.close();
        return stringsList;

    }

    private static List<String> parseTokens(String tokensStr) throws IOException {
        // You may need to adjust this according to the exact format of your serialized list
        ObjectMapper mapper = new ObjectMapper();
        return mapper.readValue(tokensStr, new TypeReference<List<String>>() {});
    }

    private static Map<Integer, List<String>> parseWordToSubtokenMap(String mapStr) throws IOException {
        // You may need to adjust this according to the exact format of your serialized map
        ObjectMapper mapper = new ObjectMapper();
        return mapper.readValue(mapStr, new TypeReference<Map<Integer, List<String>>>() {});
    }

    public static void prune(String readPath,String outPath,int flg){
//        flg=0 : codesearch , flg=1 : code summarization
        try {
            long startTime = System.currentTimeMillis();
            List<String> stringList = readFile(readPath);
//            List<String> stringList = readFile("data/codesearch/test_with_tokens.txt");
            FileOutputStream fileOutputStream = new FileOutputStream( outPath);
//            FileOutputStream fileOutputStream = new FileOutputStream( "data/codesearch/50/removed.txt");

            OutputStreamWriter outputStreamWriter = new OutputStreamWriter(fileOutputStream);
            BufferedWriter bufferedWriter = new BufferedWriter(outputStreamWriter);
//            double allRemovePercent = 0;
            int count = 0;
            for(int i=0; i < stringList.size();i++){
                id = i;
                if (i % 10000 == 0){
                    System.out.println("writing " + i + "examples");
                }
                String lineStr = stringList.get(i);
                String[] codeList = lineStr.split("<CODESPLIT>");
                String code,allTokensStr,wordToSubtokenMapStr;
                if (flg==0){
                    //code search data
                    code = codeList[0];
                    allTokensStr = codeList[3];
                    wordToSubtokenMapStr = codeList[4];
                }
                else {
                    //code summarization data
                    code = codeList[4];
                    allTokensStr = codeList[2];
                    wordToSubtokenMapStr = codeList[3];
                }

                // Deserialize all_tokens from String to List<String>
                List<String> allTokens = parseTokens(allTokensStr);

                // Deserialize word_to_subtoken_map from String to Map<Integer, Map<String, Object>>
                Map<Integer, List<String>> wordToSubtokenMap = parseWordToSubtokenMap(wordToSubtokenMapStr);
//                int originLength = code.split(" +").length;
                try {

                    code = remove(code,allTokens, wordToSubtokenMap);
                    if (flg==0){
                        codeList[0] = code;
                    }
                    else {
                        codeList[4] = code;
                    }


                    String[] newArray = new String[codeList.length - 2];
                    // 复制原数组的前(长度-2)元素到新数组
                    System.arraycopy(codeList, 0, newArray, 0, codeList.length - 2);
                    codeList = newArray;
                    String newLine = String.join("<CODESPLIT>", codeList);
                    bufferedWriter.write(newLine + "\n");
                    count++;
                }catch (ParseProblemException e){
                    System.out.println(i);
                    continue;
                }
//            System.out.println("cut:"+code.split(" +").length);
            }
            bufferedWriter.close();
            outputStreamWriter.close();
            System.out.println(count+"/"+stringList.size());
            System.out.println("共有"+count+"条数据");


            long endTime = System.currentTimeMillis();

            System.out.println("astTime:"+ astTime);
            System.out.println("labelTime:"+labelTime);
            System.out.println("removeTime:" + removeTime);
            System.out.println("totalTime:" + (endTime - startTime));
            //关闭
            if (bufferedWriter_log != null){
                bufferedWriter_log.close();
                outputStreamWriter_log.close();
                fileOutputStream_log.close();
            }

        } catch (IOException e) {
            e.printStackTrace();

        }
    }
    
    public static void main(String[] args) {
//        process test data for codesearch on codet5
        ratio=0.9;
        prune("slimcode/data/codesearch/test_with_tokens.txt","data/codesearch/slimcode/codet5/10/test.txt",1);
        ratio=0.8;
        prune("slimcode/data/codesearch/test_with_tokens.txt","data/codesearch/slimcode/codet5/20/test.txt",1);
        ratio=0.7;
        prune("slimcode/data/codesearch/test_with_tokens.txt","data/codesearch/slimcode/codet5/30/test.txt",1);
        ratio=0.6;
        prune("slimcode/data/codesearch/test_with_tokens.txt","data/codesearch/slimcode/codet5/40/test.txt",1);
        ratio=0.5;
        prune("slimcode/data/codesearch/test_with_tokens.txt","data/codesearch/slimcode/codet5/50/test.txt",1);

        //        process test data for codesearch on CodeBERT
        ratio=0.9;
        prune("slimcode/data/codesearch/test_with_tokens_codebert.txt","data/codesearch/slimcode/codebert/10/test.txt",1);
        ratio=0.8;
        prune("slimcode/data/codesearch/test_with_tokens_codebert.txt","data/codesearch/slimcode/codebert/20/test.txt",1);
        ratio=0.7;
        prune("slimcode/data/codesearch/test_with_tokens_codebert.txt","data/codesearch/slimcode/codebert/30/test.txt",1);
        ratio=0.6;
        prune("slimcode/data/codesearch/test_with_tokens_codebert.txt","data/codesearch/slimcode/codebert/40/test.txt",1);
        ratio=0.5;
        prune("slimcode/data/codesearch/test_with_tokens_codebert.txt","data/codesearch/slimcode/codebert/50/test.txt",1);

        //        process test data for code summarization on CodeT5
        ratio=0.9;
        prune("slimcode/data/code2nl/test_with_tokens.txt","data/code2nl/slimcode/codet5/10/test.txt",0);
        ratio=0.8;
        prune("slimcode/data/code2nl/test_with_tokens.txt","data/code2nl/slimcode/codet5/20/test.txt",0);
        ratio=0.7;
        prune("slimcode/data/code2nl/test_with_tokens.txt","data/code2nl/slimcode/codet5/30/test.txt",0);
        ratio=0.6;
        prune("slimcode/data/code2nl/test_with_tokens.txt","data/code2nl/slimcode/codet5/40/test.txt",0);
        ratio=0.5;
        prune("slimcode/data/code2nl/test_with_tokens.txt","data/code2nl/slimcode/codet5/50/test.txt",0);

        //        process test data for code summarization on CodeBERT
        ratio=0.9;
        prune("slimcode/data/code2nl/test_with_tokens_codebert.txt","data/code2nl/slimcode/codebert/10/test.txt",0);
        ratio=0.8;
        prune("slimcode/data/code2nl/test_with_tokens_codebert.txt","data/code2nl/slimcode/codebert/20/test.txt",0);
        ratio=0.7;
        prune("slimcode/data/code2nl/test_with_tokens_codebert.txt","data/code2nl/slimcode/codebert/30/test.txt",0);
        ratio=0.6;
        prune("slimcode/data/code2nl/test_with_tokens_codebert.txt","data/code2nl/slimcode/codebert/40/test.txt",0);
        ratio=0.5;
        prune("slimcode/data/code2nl/test_with_tokens_codebert.txt","data/code2nl/slimcode/codebert/50/test.txt",0);
    }
}
