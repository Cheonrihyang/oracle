package com.spring.view.board;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import javax.servlet.http.HttpServletRequest;

import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.ModelAndView;

import com.spring.biz.board.BoardVO;
import com.spring.biz.board.impl.BoardDAO;

@Controller
public class BoardController {
	
	
	@ModelAttribute("conditionMap")
	public Map<String,String> searchConditionMap(){
		Map<String, String> conditionMap = new HashMap<String, String>();
		conditionMap.put("제목", "TITLE");
		conditionMap.put("내용", "CONTENT");
		return conditionMap;
	}

	@RequestMapping("/getBoardList.do")
	public String getBoardList(BoardVO vo,BoardDAO boardDAO,Model model){
		if(vo.getSearchCondition()==null) vo.setSearchCondition("TITLE");
		if(vo.getSearchKeyword()==null) vo.setSearchKeyword("");
		
		model.addAttribute("boardList", boardDAO.getBoardList(vo));	//model
		return "board1/getBoardList";	//view

	}
	
	@RequestMapping("/getBoard.do")
	public String getBoard(BoardVO boardVO,BoardDAO boardDAO,Model model,HttpServletRequest req) {
		System.out.println("getBoard.do 성공");
		
		String seq = req.getParameter("seq");
		
		boardVO.setSeq(Integer.parseInt(seq));
		
		boardVO = boardDAO.getBoard(boardVO);
		
		model.addAttribute("board", boardVO);		//model
		return "board1/getBoard";	//view
	}
	
	@RequestMapping("updateBoard.do")
	public String updateBoard(HttpServletRequest req,BoardVO boardVO,BoardDAO boardDAO) {
		System.out.println("updateBoard.do 성공");
		
		boardVO.setTitle(req.getParameter("title"));
		boardVO.setContent(req.getParameter("content"));
		boardVO.setSeq(Integer.parseInt(req.getParameter("seq")));
		
		boardDAO.updateBoard(boardVO);
		
		return "redirect:getBoardList.do";
	}
	
	@RequestMapping("/deleteBoard.do")
	public String deleteBoard(HttpServletRequest req,BoardVO boardVO,BoardDAO boardDAO){
		System.out.println("deleteBoard.do 성공");
		
		
		boardVO.setSeq(Integer.parseInt(req.getParameter("seq")));
		
		boardDAO.deleteBoard(boardVO);
		
		return "redirect:getBoardList.do";
		
	}
	
	@RequestMapping("/insertBoard.do")
	public String insertBoard(ModelAndView mav,BoardVO boardVO,BoardDAO boardDAO) throws IllegalStateException, IOException{
		
		String fileName = null;
		MultipartFile uploadFile = null;
		uploadFile = boardVO.getUploadFile();
		if(uploadFile!=null) {
			fileName = uploadFile.getOriginalFilename();
			uploadFile.transferTo(new File("C:\\cha\\file\\"+fileName));
		}
		boardVO.setFileAddress("C:\\cha\\file\\"+fileName);
		
		if(boardVO.getTitle() != null) {
			
			boardDAO.insertBoard(boardVO);
			
			return "redirect:getBoardList.do";
		}else {
			return "board1/insertBoard";
		}
		
	}
}
